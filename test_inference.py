import os
import json
import re
import torch
import numpy as np
from torch import nn

# -------- utils ----------
def basic_tokenize(text: str):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text.strip())
    return text.split()

# ---- Normalizador ----
MONTHS = {
    "enero":"01","febrero":"02","marzo":"03","abril":"04","mayo":"05","junio":"06",
    "julio":"07","agosto":"08","septiembre":"09","octubre":"10","noviembre":"11","diciembre":"12"
}

def _norm_time_hhmm(m):
    h = int(m.group(1)); mm = m.group(2) or "00"
    try: mm = f"{int(mm):02d}"
    except: mm = "00"
    return f"hora {h:02d}:{mm}"

def normalize_patterns(text: str):
    s = str(text).lower().strip()
    s = re.sub(r"\borigen\s*:\s*", "origen ", s)
    s = re.sub(r"\bdestino\s*:\s*", "destino ", s)
    s = re.sub(r"\bdirección de salida\s*:\s*", "origen ", s)
    s = re.sub(r"\bla dirección es\b", ", ", s)
    if "origen" not in s and "destino" not in s:
        s = re.sub(r"^\s*(.+?)\s+hasta\s+(.+)$", r"origen \1 destino \2", s)
    s = re.sub(r"\bsalida el\s+", "", s)
    s = re.sub(
        r"\b(\d{1,2})\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)(?:\s+de\s+(\d{4}))?\b",
        lambda m: f"fecha {int(m.group(1)):02d}/{MONTHS[m.group(2)]}" + (f"/{m.group(3)}" if m.group(3) else ""),
        s
    )
    s = re.sub(
        r"(?<!fecha\s)\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b",
        lambda m: f"fecha {int(m.group(1)):02d}/{int(m.group(2)):02d}" + (f"/{m.group(3)}" if m.group(3) else ""),
        s
    )
    s = re.sub(
        r"\bregreso a las\s+(\d{1,2})(?::(\d{1,2}))?\b",
        lambda m: f"regreso hora {int(m.group(1)):02d}:{int((m.group(2) or '0')):02d}",
        s
    )
    s = re.sub(r"\bcon\s+regreso\s+(?=hora\b)", "regreso ", s)
    s = re.sub(r"\b(\d{1,2})\s*(?:h|hr|hrs|horas)\b", lambda m: f"hora {int(m.group(1)):02d}:00", s)
    s = re.sub(r"(?<!hora\s)\b(\d{1,2})[:h](\d{1,2})\b", _norm_time_hhmm, s)
    s = re.sub(r"(?<!hora\s)\ba las\s+(\d{1,2})\b", lambda m: f"hora {int(m.group(1)):02d}:00", s)
    s = re.sub(r"\ba las\s+(?=hora\b)", "", s)
    s = re.sub(r"\b(somos|para)\s+(\d{1,3})(\s*personas?)?\b", r"somos \2", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
    
    # -------- vocab y modelo ----------
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
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes=(3,4,5), num_filters=128, dropout=0.5, pad_idx=0):
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

def load_model(artifacts_dir, embed_dim=200, num_filters=128, filter_sizes=(3,4,5), pad_idx=0):
    with open(os.path.join(artifacts_dir, "vocab.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    vocab = Vocab(meta["itos"])
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    model = TextCNN(vocab_size=len(vocab), embed_dim=embed_dim, num_classes=len(id2label),
                    filter_sizes=filter_sizes, num_filters=num_filters, pad_idx=pad_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(os.path.join(artifacts_dir, "textcnn.pt"), map_location=device)
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

# -------- función de inferencia ----------
def predict_text(texts, artifacts_dir="artifacts_textcnn", max_len=160, embed_dim=200, num_filters=128, filter_sizes=(3,4,5), apply_normalizer=True):
    fsz = parse_filter_sizes(",".join(map(str, filter_sizes)))
    model, vocab, id2label, device = load_model(artifacts_dir, embed_dim, num_filters, fsz)
    
    # Normalizar
    if apply_normalizer:
        texts = [normalize_patterns(t) for t in texts]
    
    X = encode_texts(texts, vocab, max_len).to(device)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    
    results = []
    for t, pidx, p in zip(texts, preds, probs):
        results.append({
            "text": t,
            "pred_label": id2label[int(pidx)],
            "confidence": float(p[int(pidx)])
        })
    return results
 # -------- ejemplo de uso ----------
if __name__ == "__main__":
    ejemplos = [
        "hola, necesito un traslado desde providencia hasta el aeropuerto",
        "quiero cotizar un traslado para mañana",
        "solo quiero información sobre sus servicios",
        "salida el 12 de septiembre a las 14h para 4 personas"
    ]

    resultados = predict_text(ejemplos, artifacts_dir="artifacts_textcnn")

    for r in resultados:
        print(f"[{r['pred_label']}] (conf={r['confidence']:.3f}) -> {r['text']}")
