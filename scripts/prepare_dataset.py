"""
prepare_dataset.py (verboso + validador + autofix)

Uso:
  python prepare_dataset.py --input .\dataset.jsonl --output .\prepared.jsonl --ctx-turns 2 --keep-roles --autofix-labels
"""

import argparse
import json
import re
import os
from typing import List, Tuple, Dict, Any
from collections import defaultdict

RE_FECHA = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{8}|\d{1,2}\s+de\s+[a-záéíóú]+(?:\s+de\s+\d{4})?)\b",
    flags=re.IGNORECASE
)

RE_HORA = re.compile(
    r"(?<!\d)(?:\b\d{1,2}[:\.]\d{2}\b|\ba\s+las\s+\d{1,2}(?::\d{2})?\b|\b\d{3,4}\b)(?!\d)",
    flags=re.IGNORECASE
)

RE_PAX = re.compile(
    r"\b(somos|para)\s+\d+\b|\b\d+\s+(?:personas|pasajeros?)\b",
    flags=re.IGNORECASE
)

RE_REGRESO = re.compile(
    r"\b(solo\s+ida|ida\s+y\s+vuelta|con\s+regreso|sin\s+regreso)\b",
    flags=re.IGNORECASE
)

RE_ORIGEN = re.compile(
    r"\b(origen)\b|(?:\bdesde\b\s+[a-z0-9áéíóúüñ][a-z0-9áéíóúüñ\s\.\-]{3,})|(?:\bsalida\s*(?:desde|:)\s*[a-z0-9])",
    flags=re.IGNORECASE
)
RE_DESTINO = re.compile(
    r"\b(destino)\b|(?:\bhasta\b\s+[a-z0-9áéíóúüñ][a-z0-9áéíóúüñ\s\.\-]{3,})|(?:\bllegada\s*(?:a|:)\s*[a-z0-9])",
    flags=re.IGNORECASE
)

RE_FECHA_COMPACTA = re.compile(r"\b\d{8}\b")  # p.ej., 09012025

def normalize_min(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t

def split_turns(raw_text: str) -> List[Tuple[str, str]]:
    t = raw_text
    t = t.replace("<usr>", "\n<usr>").replace("<sys>", "\n<sys>")
    chunks = [c for c in t.split("\n") if c.strip()]
    turns: List[Tuple[str, str]] = []
    for c in chunks:
        c = c.strip()
        if c.startswith("<usr>"):
            turns.append(("USR", c[len("<usr>"):].strip()))
        elif c.startswith("<sys>"):
            turns.append(("SYS", c[len("<sys>"):].strip()))
        else:
            turns.append(("USR", c))
    return turns

def build_context(turns: List[Tuple[str, str]], ctx_turns: int) -> Tuple[str, str, List[Tuple[str, str]]]:
    if not turns:
        return "", "", []

    idx_last_usr = None
    for i in range(len(turns) - 1, -1, -1):
        if turns[i][0] == "USR":
            idx_last_usr = i
            break
    last_idx = idx_last_usr if idx_last_usr is not None else len(turns) - 1

    cur_role, cur_msg = turns[last_idx]
    start_ctx = max(0, last_idx - ctx_turns)
    ctx_slice = turns[start_ctx:last_idx]

    parts = []
    for r, m in ctx_slice:
        if m:
            parts.append(f"[{r}] {normalize_min(m)}")
    parts.append(f"[{cur_role}] {normalize_min(cur_msg)}")
    full_text = " [SEP] ".join(parts)

    return full_text, normalize_min(cur_msg), ctx_slice

def only_user_text_from_concat(full_text_with_roles: str) -> str:
    pieces = []
    for chunk in full_text_with_roles.split("[SEP]"):
        c = chunk.strip()
        if c.startswith("[USR]"):
            pieces.append(c[len("[USR]"):].strip())
    return " ".join(pieces).strip()

def mask_fecha_compacta(text: str) -> str:
    return RE_FECHA_COMPACTA.sub(" FECHACOMPACTA ", text)

def detect_slots(text_user_only: str) -> List[int]:
    t = mask_fecha_compacta(text_user_only)
    has_origen = 1 if RE_ORIGEN.search(t) else 0
    has_destino = 1 if RE_DESTINO.search(t) else 0
    has_fecha = 1 if RE_FECHA.search(t) else 0
    has_hora = 1 if RE_HORA.search(t) else 0
    has_pax = 1 if RE_PAX.search(t) else 0
    has_regreso = 1 if RE_REGRESO.search(t) else 0
    return [has_origen, has_destino, has_fecha, has_hora, has_pax, has_regreso]

def truncate_tokens(text: str, max_len: int) -> str:
    toks = text.split()
    if len(toks) <= max_len:
        return text
    return " ".join(toks[-max_len:])

def expected_label_from_state(state_count: int) -> str:
    if state_count >= 6:
        return "Cotización generada"
    if state_count <= 0:
        return "Potencial cliente"
    return "Cotizando"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Ruta del dataset JSONL original")
    ap.add_argument("--output", required=False, help="Ruta del JSONL preparado")
    ap.add_argument("--ctx-turns", type=int, default=2, help="Turnos previos a incluir (default=2)")
    ap.add_argument("--max-len", type=int, default=320, help="Máx. tokens tras concatenar (default=320)")
    ap.add_argument("--keep-roles", action="store_true",
                    help="Conservar [USR]/[SYS] en el texto (recomendado para prod).")
    ap.add_argument("--validate-only", action="store_true",
                    help="Solo validar y reportar inconsistencias; no escribir archivo de salida.")
    ap.add_argument("--autofix-labels", action="store_true",
                    help="Sobrescribe 'label' según slots_state_count y guarda 'label_original'.")
    ap.add_argument("--quiet", action="store_true", help="Menos logs.")
    args = ap.parse_args()

    if not args.validate_only and not args.output:
        raise SystemExit("Error: --output es requerido excepto si usas --validate-only")

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output) if args.output else None

    if not args.quiet:
        print(f"[prepare] Python: {os.sys.version.split()[0]}")
        print(f"[prepare] Input : {input_path}")
        if output_path:
            print(f"[prepare] Output: {output_path}")
        print(f"[prepare] ctx_turns={args.ctx_turns} keep_roles={args.keep_roles} max_len={args.max_len}")
        print(f"[prepare] validate_only={args.validate_only} autofix_labels={args.autofix_labels}")

    rows: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            obj["_orig_idx"] = idx
            rows.append(obj)

    if not args.quiet:
        print(f"[prepare] Leídas {len(rows)} líneas del input.")

    chats: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for obj in rows:
        chats[str(obj.get("chat_id"))].append(obj)

    for chat_id in chats:
        chats[chat_id].sort(key=lambda o: (o.get("msg_idx", 0), o["_orig_idx"]))

    outputs: List[Dict[str, Any]] = []
    total_inconsist = 0

    for chat_id, items in chats.items():
        cumulative_user_text = ""

        for obj in items:
            raw_text = obj.get("text", "")
            turns = split_turns(raw_text)

            text_ctx, cur_msg, ctx_slice = build_context(turns, args.ctx_turns)

            final_text = text_ctx
            if not args.keep_roles:
                final_text = final_text.replace("[USR]", "").replace("[SYS]", "").strip()
                final_text = re.sub(r"\s+\[SEP\]\s+", " [SEP] ", final_text)

            final_text = truncate_tokens(final_text, args.max_len)

            user_only_window = only_user_text_from_concat(text_ctx)
            user_only_window = normalize_min(user_only_window)
            slots = detect_slots(user_only_window)
            slots_count = int(sum(slots))

            cumulative_user_text = normalize_min((cumulative_user_text + " " + user_only_window).strip())
            slots_state = detect_slots(cumulative_user_text)
            slots_state_count = int(sum(slots_state))

            label = obj.get("label")
            exp_label = expected_label_from_state(slots_state_count)
            is_inconsistent = (label != exp_label)

            out = {
                "id": obj.get("id"),
                "chat_id": obj.get("chat_id"),
                "msg_idx": obj.get("msg_idx"),
                "text": final_text,
                "slots": slots,
                "slots_count": slots_count,
                "slots_state": slots_state,
                "slots_state_count": slots_state_count,
                "label": label,
                "label_expected": exp_label,
                "label_consistent": not is_inconsistent,
                "_orig_idx": obj["_orig_idx"],
            }

            if is_inconsistent and args.autofix_labels:
                out["label_original"] = label
                out["label"] = exp_label

            if is_inconsistent:
                total_inconsist += 1

            outputs.append(out)

    total = len(outputs)
    pct = (total_inconsist / total * 100.0) if total else 0.0
    if not args.quiet:
        print(f"[prepare] Inconsistencias: {total_inconsist} / {total}  ({pct:.2f}%)")

    if args.validate_only:
        if not args.quiet:
            shown = 0
            print("[prepare] Muestras de inconsistencias (máx 20):")
            for o in sorted(outputs, key=lambda x: x["_orig_idx"]):
                if not o["label_consistent"]:
                    if shown < 20:
                        print(json.dumps({
                            "id": o["id"],
                            "chat_id": o["chat_id"],
                            "msg_idx": o["msg_idx"],
                            "label": o["label"],
                            "label_expected": o["label_expected"],
                            "slots_state_count": o["slots_state_count"]
                        }, ensure_ascii=False))
                        shown += 1
        return

    outputs.sort(key=lambda o: o["_orig_idx"])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for o in outputs:
            o.pop("_orig_idx", None)
            fout.write(json.dumps(o, ensure_ascii=False) + "\n")

    if not args.quiet:
        print(f"[prepare] Procesadas {len(rows)} líneas. Escritas {len(outputs)} en {output_path}")
        if args.autofix_labels:
            print("[prepare] --autofix-labels aplicado cuando label != label_expected.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
