import json
import sys
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_file", type=str, default=None, 
                       help="Ruta al archivo de métricas; si no se pasa, se usa <artifacts_dir>/metrics.json")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts_textcnn_aug10k",
                       help="Directorio de artifacts donde buscar metrics.json si no se especifica --metrics_file")
    parser.add_argument("--min_accuracy", type=float, default=0.96,
                       help="Accuracy mínimo para aprobar el modelo (default: 0.96)")
    parser.add_argument("--max_accuracy", type=float, default=1.0,
                       help="Accuracy máximo para aprobar el modelo (default: 1.0)")
    
    args = parser.parse_args()

    metrics_path = args.metrics_file or os.path.join(args.artifacts_dir, "metrics.json")

    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print(f"No se encontró {metrics_path}. ¿Ejecutaste primero train.py?")
        sys.exit(1)

    accuracy = metrics.get("accuracy", 0)
    f1 = metrics.get("f1", 0)

    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    if accuracy < args.min_accuracy or accuracy > args.max_accuracy:
        print(f"Modelo rechazado: accuracy ({accuracy:.4f}) fuera de rango [{args.min_accuracy}, {args.max_accuracy}]")
        sys.exit(1)
    else:
        print("Modelo aprobado")
        sys.exit(0)

if __name__ == "__main__":
    main()
