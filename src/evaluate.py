import json
import sys

# Cargar métricas desde metrics.json
try:
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
except FileNotFoundError:
    print("No se encontró metrics.json. ¿Ejecutaste primero train.py?")
    sys.exit(1)

accuracy = metrics.get("accuracy", 0)
f1 = metrics.get("f1", 0)

print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# Reglas de aprobación
if accuracy < 0.96 or accuracy == 1.0:
    print("Modelo rechazado: accuracy fuera de rango")
    sys.exit(1)
else:
    print("Modelo aprobado")
    sys.exit(0)
