# evaluar_progreso.py
import re
import torch
import numpy as np

# Utilidades del predictor (usa tu versión con normalize_patterns)
from predict_textcnn import load_model, encode_texts, normalize_patterns, parse_filter_sizes

# ================== Config ==================
ARTIFACTS_DIR = "artifacts_textcnn"
EMBED_DIM = 200
NUM_FILTERS = 128
FILTER_SIZES_STR = "2,3,4,5"
MAX_LEN = 256

APPLY_NORMALIZER = True
ENFORCE_FIELDS = True   # pon en False para evaluar solo la red

# Campos obligatorios (para ENFORCE_FIELDS)
REQ = {
    "origen": re.compile(r"\borigen\b"),
    "destino": re.compile(r"\bdestino\b"),
    "fecha": re.compile(r"\bfecha\b"),
    "hora": re.compile(r"\bhora\b"),
    "cantidad": re.compile(r"\bsomos\s+\d{1,3}\b"),
}

# ================== Carga modelo ==================
fsz = parse_filter_sizes(FILTER_SIZES_STR)
model, vocab, id2label, device = load_model(
    ARTIFACTS_DIR, embed_dim=EMBED_DIM, num_filters=NUM_FILTERS, filter_sizes=fsz
)
label2id = {v: k for k, v in id2label.items()}

# ================== Reglas extra de normalización ==================
def apply_alias_rules(s: str) -> str:
    """
    Reglas complementarias (idempotentes) para capturar variantes comunes.
    Se asume que 's' ya está en minúsculas (normalize_patterns hace lower()).
    """
    # Etiquetas con ":", por si quedan
    s = re.sub(r"\borigen\s*:\s*", "origen ", s)
    s = re.sub(r"\bdestino\s*:\s*", "destino ", s)
    s = re.sub(r"\bdirección de salida\s*:\s*", "origen ", s)

    # "desde ... hasta ..." -> origen/destino (también si aparece tras <sep>)
    s = re.sub(r"(^|<sep>\s*)desde\s+", r"\1origen ", s)
    s = re.sub(r"\s+hasta\s+", " destino ", s)

    # ORIGEN: frases típicas
    s = re.sub(r"\bla idea es\s+comenzar en\s+", "origen ", s)
    s = re.sub(r"\b(partir|salir)\s+desde\s+", "origen ", s)
    s = re.sub(r"\b(iniciar|inicio)\s+en\s+", "origen ", s)

    # DESTINO: frases típicas
    s = re.sub(r"\b(nuestro objetivo es\s+)?(llegar a|dirigirse a|ir a|ir hasta|terminar en|finalizar en)\s+", "destino ", s)

    # CANTIDAD: "viajamos N ..." -> "somos N"
    s = re.sub(r"\bviajamos\s+(\d{1,3})(\s*personas?)?(\s+en\s+total)?\b", r"somos \1", s)

    return s

def normalize_history_and_join(history_msgs, sep="<SEP>"):
    """
    Normaliza cada mensaje con normalize_patterns, concatena con <SEP>,
    y aplica reglas extra sobre el combinado (en minúsculas).
    Devuelve el texto final que realmente se clasifica.
    """
    if APPLY_NORMALIZER:
        norm_segments = [normalize_patterns(m) for m in history_msgs]
    else:
        norm_segments = history_msgs[:]

    joined = f" {sep} ".join(norm_segments)
    # aplicar reglas extra sobre versión minúscula para detectar "<sep>" y patrones amplios
    joined_final = apply_alias_rules(joined.lower())
    return joined_final

# ================== Helpers ==================
def apply_enforce_fields(texto_norm_joined: str, pred_idx: int) -> int:
    if not ENFORCE_FIELDS:
        return pred_idx
    pred_lbl = id2label[int(pred_idx)]
    REQ = {
        "origen": re.compile(r"\borigen\b"),
        "destino": re.compile(r"\bdestino\b"),
        "fecha": re.compile(r"\bfecha\b"),
        "hora": re.compile(r"\bhora\b"),
        "cantidad": re.compile(r"\bsomos\s+\d{1,3}\b"),
    }
    MIN_FIELDS_FOR_QUOTE = 2

    flags = {k: bool(rx.search(texto_norm_joined)) for k, rx in REQ.items()}
    covered = sum(flags.values())

    if covered == len(REQ):
        return label2id.get("Cotización generada", pred_idx)
    if covered >= MIN_FIELDS_FOR_QUOTE and id2label[pred_idx] == "Potencial cliente":
        return label2id.get("Cotizando", pred_idx)
    if covered < len(REQ) and id2label[pred_idx] == "Cotización generada":
        return label2id.get("Cotizando", pred_idx)
    return pred_idx


def predict_history(history_msgs):
    texto = normalize_history_and_join(history_msgs)  # string final usado por el modelo
    X = encode_texts([texto], vocab, MAX_LEN).to(device)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(logits.argmax(dim=1).cpu().numpy()[0])
    final_idx = apply_enforce_fields(texto, pred_idx)
    return id2label[pred_idx], float(probs[pred_idx]), id2label[final_idx], float(probs[final_idx]), texto

# ================== Evaluación ==================
def evaluar_modelo(test_cases):
    total_steps, total_hits = 0, 0

    for case_idx, caso in enumerate(test_cases):
        print(f"\n================  CASO {case_idx+1}  ================")
        history = []
        steps_hits = 0

        for step_idx, (texto, esperado) in enumerate(zip(caso["texto"], caso["esperado"]), start=1):
            history.append(texto)
            model_pred, model_conf, final_pred, final_conf, texto_final = predict_history(history)

            # Usamos la predicción final (con reglas si ENFORCE_FIELDS=True)
            pred_lbl = final_pred
            conf = final_conf
            ok = (pred_lbl == esperado)

            steps_hits += int(ok)
            total_hits += int(ok)
            total_steps += 1

            check = "✓" if ok else "✗"
            print(f"Paso {step_idx:02d}: pred={pred_lbl:>19s}  | esperado={esperado:>19s}   {check}")

            # Debug condicional: si falla, mostramos el texto normalizado final que se clasificó
            if not ok:
                print("DEBUG texto_norm_joined:")
                # Volvemos a imprimir con <SEP> visible (ya está en minúsculas por join+lower)
                print(texto_final)

        acc = steps_hits / len(caso["texto"])
        print(f"Resumen CASO {case_idx+1}: {steps_hits}/{len(caso['texto'])} correctos (acc={acc:.3f})")

    overall = total_hits / total_steps if total_steps else 0.0
    print(f"\n================  RESUMEN GLOBAL  ================")
    print(f"Aciertos totales: {total_hits}/{total_steps}  (acc={overall:.3f})")

if __name__ == "__main__":
    # Debes tener un archivo test_cases.py con la variable `test_cases`
    from casos_test import test_cases
    evaluar_modelo(test_cases)
