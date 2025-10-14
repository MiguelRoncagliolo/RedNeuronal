import json
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_file", type=str, default="metrics.json", 
                       help="Ruta al archivo de métricas (default: metrics.json)")
    parser.add_argument("--min_accuracy", type=float, default=0.96,
                       help="Accuracy mínimo para aprobar el modelo (default: 0.96)")
    parser.add_argument("--max_accuracy", type=float, default=1.0,
                       help="Accuracy máximo para aprobar el modelo (default: 1.0)")
    
    args = parser.parse_args()

    # Cargar métricas desde metrics.json
    try:
        with open(args.metrics_file, "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print(f"No se encontró {args.metrics_file}. ¿Ejecutaste primero train.py?")
        sys.exit(1)

    accuracy = metrics.get("accuracy", 0)
    f1 = metrics.get("f1", 0)

    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    # Reglas de aprobación
    if accuracy < args.min_accuracy or accuracy >= args.max_accuracy:
        print(f"Modelo rechazado: accuracy ({accuracy:.4f}) fuera de rango [{args.min_accuracy}, {args.max_accuracy})")
        sys.exit(1)
    else:
        print("Modelo aprobado")
        sys.exit(0)

if __name__ == "__main__":
    main()
