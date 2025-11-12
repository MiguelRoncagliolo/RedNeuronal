import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict_text import load_model, encode_texts, parse_filter_sizes
from normalizer import normalize_history_and_join, apply_enforce_fields

ARTIFACTS_DIR = "../artifacts_textcnn"
EMBED_DIM = 200
NUM_FILTERS = 128
FILTER_SIZES_STR = "3,4,5"
MAX_LEN = 256

APPLY_NORMALIZER = True
ENFORCE_FIELDS = True
fsz = parse_filter_sizes(FILTER_SIZES_STR)
model, vocab, id2label, device = load_model(
    ARTIFACTS_DIR, embed_dim=EMBED_DIM, num_filters=NUM_FILTERS, filter_sizes=fsz
)
label2id = {v: k for k, v in id2label.items()}

def predict_history(history_msgs):
    texto = normalize_history_and_join(history_msgs, sep="<SEP>", apply_normalizer=APPLY_NORMALIZER)
    X = encode_texts([texto], vocab, MAX_LEN).to(device)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(logits.argmax(dim=1).cpu().numpy()[0])
    final_idx = apply_enforce_fields(
        texto_norm_joined=texto,
        pred_idx=pred_idx,
        id2label=id2label,
        label2id=label2id,
        enforce_fields=ENFORCE_FIELDS,
    )
    return id2label[pred_idx], float(probs[pred_idx]), id2label[final_idx], float(probs[final_idx]), texto

def evaluar_modelo(test_cases):
    total_steps, total_hits = 0, 0
    errores = []

    for case_idx, caso in enumerate(test_cases):
        print(f"\n================  CASO {case_idx+1}  ================")
        history = []
        steps_hits = 0

        for step_idx, (texto, esperado) in enumerate(zip(caso["texto"], caso["esperado"]), start=1):
            history.append(texto)
            model_pred, model_conf, final_pred, final_conf, texto_final = predict_history(history)

            pred_lbl = final_pred
            conf = final_conf
            ok = (pred_lbl == esperado)

            steps_hits += int(ok)
            total_hits += int(ok)
            total_steps += 1

            check = "[OK]" if ok else "[FAIL]"
            print(f"Paso {step_idx:02d}: pred={pred_lbl:>19s}  | esperado={esperado:>19s}   {check}")

            if not ok:
                print("DEBUG texto_norm_joined:")
                print(texto_final)
                errores.append({
                    'caso': case_idx + 1,
                    'paso': step_idx,
                    'mensaje_original': texto,
                    'prediccion': pred_lbl,
                    'esperado': esperado,
                    'texto_normalizado': texto_final,
                    'confianza_modelo': model_conf,
                    'confianza_final': final_conf
                })

        acc = steps_hits / len(caso["texto"])
        print(f"Resumen CASO {case_idx+1}: {steps_hits}/{len(caso['texto'])} correctos (acc={acc:.3f})")

    overall = total_hits / total_steps if total_steps else 0.0
    print(f"\n================  RESUMEN GLOBAL  ================")
    print(f"Aciertos totales: {total_hits}/{total_steps}  (acc={overall:.3f})")
    
    if errores:
        print(f"\n================  LOG COMPLETO DE ERRORES ({len(errores)} errores)  ================")
        for i, error in enumerate(errores, 1):
            print(f"\nERROR #{i}:")
            print(f"   Caso {error['caso']}, Paso {error['paso']}")
            print(f"   Mensaje: '{error['mensaje_original']}'")
            print(f"   Prediccion: {error['prediccion']} (conf: {error['confianza_final']:.3f})")
            print(f"   Esperado: {error['esperado']}")
            print(f"   Texto normalizado: {error['texto_normalizado']}")
            
        print(f"\nRESUMEN DE ERRORES:")
        tipos_error = {}
        for error in errores:
            key = f"{error['esperado']} ← {error['prediccion']}"
            tipos_error[key] = tipos_error.get(key, 0) + 1
        
        print("   Tipos de error mas comunes:")
        for tipo, count in sorted(tipos_error.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {tipo}: {count} casos")
            
        casos_errores = {}
        for error in errores:
            casos_errores[error['caso']] = casos_errores.get(error['caso'], 0) + 1
        
        if len(casos_errores) < len(test_cases):
            print(f"\n   Casos mas problematicos:")
            for caso, count in sorted(casos_errores.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   - Caso {caso}: {count} errores")
    else:
        print(f"\nPERFECTO: No hay errores en la evaluacion.")

if __name__ == "__main__":
    from test_cases import test_cases
    evaluar_modelo(test_cases)