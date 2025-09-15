# evaluar_progreso.py
import re
import torch
import numpy as np

# Utilidades del predictor y normalizador unificado
from predict_textcnn import load_model, encode_texts, parse_filter_sizes
from normalizer import normalize_history_and_join, apply_enforce_fields

# ================== Config ==================
ARTIFACTS_DIR = "artifacts_textcnn"
EMBED_DIM = 200
NUM_FILTERS = 128
FILTER_SIZES_STR = "2,3,4,5"
MAX_LEN = 256

APPLY_NORMALIZER = True
ENFORCE_FIELDS = True   # pon en False para evaluar solo la red

# Los campos obligatorios ahora están en normalizer.py

# ================== Carga modelo ==================
fsz = parse_filter_sizes(FILTER_SIZES_STR)
model, vocab, id2label, device = load_model(
    ARTIFACTS_DIR, embed_dim=EMBED_DIM, num_filters=NUM_FILTERS, filter_sizes=fsz
)
label2id = {v: k for k, v in id2label.items()}

# Las funciones de normalización están ahora en normalizer.py

# Los helpers están ahora en normalizer.py


def predict_history(history_msgs):
    texto = normalize_history_and_join(history_msgs, sep="<SEP>", apply_normalizer=APPLY_NORMALIZER)  # string final usado por el modelo
    X = encode_texts([texto], vocab, MAX_LEN).to(device)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(logits.argmax(dim=1).cpu().numpy()[0])
    final_idx = apply_enforce_fields(texto, pred_idx, id2label, label2id, 
                                   enforce_fields=ENFORCE_FIELDS, min_fields_for_quote=2)
    return id2label[pred_idx], float(probs[pred_idx]), id2label[final_idx], float(probs[final_idx]), texto

# ================== Evaluación ==================
def evaluar_modelo(test_cases):
    total_steps, total_hits = 0, 0
    errores = []  # Lista para almacenar todos los errores

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
                
                # Guardar error para log final
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
    
    # ================== LOG DE ERRORES ==================
    if errores:
        print(f"\n================  LOG COMPLETO DE ERRORES ({len(errores)} errores)  ================")
        for i, error in enumerate(errores, 1):
            print(f"\n❌ ERROR #{i}:")
            print(f"   📍 Caso {error['caso']}, Paso {error['paso']}")
            print(f"   💬 Mensaje: '{error['mensaje_original']}'")
            print(f"   🤖 Predicción: {error['prediccion']} (conf: {error['confianza_final']:.3f})")
            print(f"   ✅ Esperado: {error['esperado']}")
            print(f"   🔧 Texto normalizado: {error['texto_normalizado']}")
            
        print(f"\n📊 RESUMEN DE ERRORES:")
        # Contar errores por tipo de transición
        tipos_error = {}
        for error in errores:
            key = f"{error['esperado']} ← {error['prediccion']}"
            tipos_error[key] = tipos_error.get(key, 0) + 1
        
        print("   Tipos de error más comunes:")
        for tipo, count in sorted(tipos_error.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {tipo}: {count} casos")
            
        # Casos más problemáticos
        casos_errores = {}
        for error in errores:
            casos_errores[error['caso']] = casos_errores.get(error['caso'], 0) + 1
        
        if len(casos_errores) < len(test_cases):
            print(f"\n   Casos más problemáticos:")
            for caso, count in sorted(casos_errores.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   • Caso {caso}: {count} errores")
    else:
        print(f"\n🎉 ¡PERFECTO! No hay errores en la evaluación.")

if __name__ == "__main__":
    # Debes tener un archivo test_cases.py con la variable `test_cases`
    from casos_test import test_cases
    evaluar_modelo(test_cases)
