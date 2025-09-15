import argparse
import json
import os
import re
import torch
import numpy as np
from torch import nn

# -------- utils ----------
def basic_tokenize(text: str):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text.strip())
    return text.split()

# Importar normalizador unificado
from normalizer import normalize_patterns



class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {t:i for i,t in enumerate(self.itos)}
    def encode(self, toks):
        unk = self.stoi.get("<unk>", 1)
        return [self.stoi.get(t, unk) for t in toks]
    def __len__(self):
        return len(self.itos)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes=(2,3,4,5), num_filters=128, dropout=0.5, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        conv_outs = [torch.relu(conv(emb)) for conv in self.convs]
        pooled = [torch.max(co, dim=2).values for co in conv_outs]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

def parse_filter_sizes(s: str):
    try:
        return tuple(int(x.strip()) for x in s.split(",") if x.strip())
    except Exception:
        return (3,4,5)

def load_model(artifacts_dir, embed_dim=200, num_filters=128, filter_sizes=(2,3,4,5), pad_idx=0):
    with open(os.path.join(artifacts_dir, "vocab.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    vocab = Vocab(meta["itos"])
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    model = TextCNN(vocab_size=len(vocab), embed_dim=embed_dim, num_classes=len(id2label),
                    filter_sizes=filter_sizes, num_filters=num_filters, pad_idx=pad_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(os.path.join(artifacts_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, vocab, id2label, device

def encode_texts(texts, vocab, max_len=160):
    batch = []
    for text in texts:
        toks = basic_tokenize(text)
        ids = vocab.encode(toks)[:max_len]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        batch.append(ids)
    return torch.tensor(batch, dtype=torch.long)

# -------- main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", nargs="+", help="Uno o varios textos a clasificar (si no usas --history)")
    parser.add_argument("--history", nargs="+", help="Mensajes previos en orden (se concatenan con --context_sep)")
    parser.add_argument("--context_sep", type=str, default="<SEP>")
    parser.add_argument("--apply_normalizer", action="store_true",
                        help="Aplica las mismas normalizaciones que en train")
    parser.add_argument("--enforce_fields", action="store_true",
                        help="Si falta origen/destino/fecha/hora/cantidad, fuerza 'Cotizando'")
    parser.add_argument("--debug_norm", action="store_true",
                        help="Imprime el texto ya normalizado antes de clasificar")
    parser.add_argument("--artifacts", default="artifacts_textcnn")
    parser.add_argument("--max_len", type=int, default=160)
    parser.add_argument("--embed_dim", type=int, default=200)
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--filter_sizes", type=str, default="2,3,4,5", help='Ej: "2,3,4,5"')
    args = parser.parse_args()

    # Construir el texto final
    if args.history:
        joined = f" {args.context_sep} ".join(args.history)
        texts = [joined]
    elif args.text:
        texts = args.text
    else:
        parser.error("Debes pasar --text '...' o --history 'm1' 'm2' ...")

    # Normalizar por segmento (para no cruzar el <SEP>)
    if args.apply_normalizer:
        normed = []
        for t in texts:
            parts = [p.strip() for p in t.split(args.context_sep)]
            parts = [normalize_patterns(p) for p in parts]
            normed.append(f" {args.context_sep} ".join(parts))
        texts = normed

    if args.debug_norm:
        print("=== NORMALIZED ===")
        for t in texts:
            print(t)

    fsz = parse_filter_sizes(args.filter_sizes)
    model, vocab, id2label, device = load_model(
        args.artifacts, embed_dim=args.embed_dim, num_filters=args.num_filters, filter_sizes=fsz
    )

    X = encode_texts(texts, vocab, args.max_len).to(device)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    # Regla opcional: exigir campos usando normalizador unificado
    if args.enforce_fields:
        from normalizer import apply_enforce_fields
        label2id_local = {v: k for k, v in id2label.items()}
        fixed = []
        for t, pidx in zip(texts, preds):
            fixed_idx = apply_enforce_fields(t, pidx, id2label, label2id_local, 
                                           enforce_fields=True, min_fields_for_quote=2)
            fixed.append(fixed_idx)
        preds = np.array(fixed)

    for t, pidx, p in zip(texts, preds, probs):
        print(f"[{id2label[int(pidx)]}] (conf={float(p[int(pidx)]):.3f}) -> {t}")
