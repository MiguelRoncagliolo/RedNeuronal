"""
TextCNN training script (multi-turn ready) for chat_user_turns_v4_5000.csv
"""

import argparse
import json
import os
import random
import re
from collections import Counter

import numpy as np

# Try to import torch/sklearn; give a friendly error if unavailable
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    raise RuntimeError("Este script requiere PyTorch. Instala con: pip install torch") from e

try:
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
except Exception as e:
    raise RuntimeError("Este script requiere scikit-learn. Instala con: pip install scikit-learn") from e

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("Este script requiere pandas. Instala con: pip install pandas") from e


# -------------------- utils --------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def basic_tokenize(text: str):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text.strip())
    return text.split()


# Importar normalizador unificado
from normalizer import normalize_patterns


class Vocab:
    def __init__(self, min_freq: int = 2, specials=None, max_size: int = None):
        self.freqs = Counter()
        self.itos = []
        self.stoi = {}
        self.min_freq = min_freq
        self.max_size = max_size
        self.specials = specials or ["<pad>", "<unk>"]

    def build(self, token_lists):
        for toks in token_lists:
            self.freqs.update(toks)
        self.itos = list(self.specials)
        for i, sp in enumerate(self.itos):
            self.stoi[sp] = i
        items = sorted(self.freqs.items(), key=lambda kv: (-kv[1], kv[0]))
        for tok, f in items:
            if f < self.min_freq:
                continue
            if self.max_size and len(self.itos) >= self.max_size:
                break
            if tok in self.stoi:
                continue
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)

    def encode(self, toks):
        unk = self.stoi["<unk>"]
        return [self.stoi.get(t, unk) for t in toks]

    def __len__(self):
        return len(self.itos)


class TextDataset(Dataset):
    def __init__(self, df, text_col, label_col, vocab, label2id, max_len):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.label_col = label_col
        self.vocab = vocab
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        toks = basic_tokenize(row[self.text_col])
        ids = self.vocab.encode(toks)[: self.max_len]
        pad_id = 0
        if len(ids) < self.max_len:
            ids = ids + [pad_id] * (self.max_len - len(ids))
        label = self.label2id[row[self.label_col]]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes=(3,4,5), num_filters=100, dropout=0.5, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)     # (B, E, L)
        conv_outs = [torch.relu(c(emb)) for c in self.convs]
        pooled = [torch.max(co, dim=2).values for co in conv_outs]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


def compute_class_weights(labels, label2id):
    from collections import Counter
    counts = Counter(labels)
    n = sum(counts.values())
    weights = {}
    for lbl, idx in label2id.items():
        c = counts.get(lbl, 1)
        weights[idx] = n / (len(label2id) * c)
    return torch.tensor([weights[i] for i in range(len(label2id))], dtype=torch.float)


def split_by_group(df, label_col, group_col="chat_id", seed=42):
    groups = df[group_col].astype(str).values
    labels = df[label_col].astype(str).values
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=seed)
    train_idx, rest_idx = next(gss1.split(X=df, y=labels, groups=groups))
    df_train = df.iloc[train_idx].copy()
    df_rest = df.iloc[rest_idx].copy()
    groups_rest = df_rest[group_col].astype(str).values
    labels_rest = df_rest[label_col].astype(str).values
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed+1)
    val_idx, test_idx = next(gss2.split(X=df_rest, y=labels_rest, groups=groups_rest))
    df_val = df_rest.iloc[val_idx].copy()
    df_test = df_rest.iloc[test_idx].copy()
    return df_train, df_val, df_test


def build_context_windows(df, text_col, group_col, k, sep):
    if k <= 0:
        df["context_text"] = df[text_col].astype(str)
        return df
    if "msg_idx" not in df.columns:
        raise ValueError("Para contexto multi-turno necesitas columna 'msg_idx' para ordenar los mensajes.")
    out_rows = []
    for chat_id, grp in df.groupby(group_col):
        grp = grp.sort_values("msg_idx").copy()
        history = []
        for _, row in grp.iterrows():
            ctx_msgs = history[-k:] + [str(row[text_col])]
            ctx = f" {sep} ".join([m for m in ctx_msgs if m])
            row_out = row.to_dict()
            row_out["context_text"] = ctx
            out_rows.append(row_out)
            history.append(str(row[text_col]))
    return pd.DataFrame(out_rows)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for X, y in loader:
        X = X.to(device); y = y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * X.size(0)
        all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for X, y in loader:
        X = X.to(device); y = y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1, all_labels, all_preds


def parse_filter_sizes(s: str):
    try:
        return tuple(int(x.strip()) for x in s.split(",") if x.strip())
    except Exception:
        return (3, 4, 5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--text_col", type=str, default="window_text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--group_col", type=str, default="chat_id")
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_size", type=int, default=50000)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=200)
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--filter_sizes", type=str, default="3,4,5")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./artifacts_textcnn")
    parser.add_argument("--disable_class_weights", action="store_true")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--disable_early_stop", action="store_true")
    parser.add_argument("--context_window", type=int, default=0)
    parser.add_argument("--context_sep", type=str, default="<SEP>")
    parser.add_argument("--normalize_patterns", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    df = df.dropna(subset=[args.text_col, args.label_col]).copy()

    # Normaliza por mensaje ANTES de construir ventanas de contexto
    if args.normalize_patterns:
        df[args.text_col] = df[args.text_col].astype(str).apply(normalize_patterns)

    df = build_context_windows(df, args.text_col, args.group_col, args.context_window, args.context_sep)
    used_text_col = "context_text" if args.context_window > 0 else args.text_col

    labels = sorted(df[args.label_col].astype(str).unique().tolist())
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    df["label_id"] = df[args.label_col].map(label2id)

    token_lists = [basic_tokenize(t) for t in df[used_text_col].astype(str).tolist()]
    vocab = Vocab(min_freq=args.min_freq, max_size=args.max_size)
    vocab.build(token_lists)

    if args.group_col in df.columns:
        train_df, val_df, test_df = split_by_group(df, args.label_col, args.group_col, args.seed)
    else:
        train_df, rest_df = train_test_split(df, test_size=0.3, random_state=args.seed, stratify=df[args.label_col])
        val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=args.seed, stratify=rest_df[args.label_col])

    train_ds = TextDataset(train_df, used_text_col, args.label_col, vocab, label2id, args.max_len)
    val_ds   = TextDataset(val_df,   used_text_col, args.label_col, vocab, label2id, args.max_len)
    test_ds  = TextDataset(test_df,  used_text_col, args.label_col, vocab, label2id, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fsz = parse_filter_sizes(args.filter_sizes)

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        num_classes=len(label2id),
        filter_sizes=fsz,
        num_filters=args.num_filters,
        dropout=args.dropout,
        pad_idx=0
    ).to(device)

    if args.disable_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        class_weights = compute_class_weights(train_df[args.label_col].astype(str).tolist(), label2id).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    patience = args.patience
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1, _, _ = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d} | train_loss {tr_loss:.4f} | train_f1 {tr_f1:.4f} | val_loss {val_loss:.4f} | val_f1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            bad_epochs = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "textcnn.pt"))
            with open(os.path.join(args.save_dir, "vocab.json"), "w", encoding="utf-8") as f:
                json.dump({"itos": vocab.itos, "label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
        else:
            bad_epochs += 1
            if not args.disable_early_stop and bad_epochs >= patience:
                print("Early stopping.")
                break

    best_path = os.path.join(args.save_dir, "textcnn.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_loss, test_f1, y_true, y_pred = eval_epoch(model, test_loader, criterion, device)
    print("\n===== TEST RESULTS =====")
    print(f"test_loss {test_loss:.4f} | test_f1 {test_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(id2label))], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    with open(os.path.join(args.save_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "TextCNN artifacts\n"
            f"vocab_size={len(vocab)}\n"
            f"embed_dim={args.embed_dim}\n"
            f"num_filters={args.num_filters}\n"
            f"filter_sizes={fsz}\n"
            f"dropout={args.dropout}\n"
            f"max_len={args.max_len}\n"
            f"context_window={args.context_window}\n"
            f"context_sep={args.context_sep}\n"
            f"weight_decay={args.weight_decay}\n"
            f"normalize_patterns={args.normalize_patterns}\n"
            f"epochs={epoch}\n"
            f"best_val_f1={best_val_f1:.4f}\n"
        )


if __name__ == "__main__":
    main()
