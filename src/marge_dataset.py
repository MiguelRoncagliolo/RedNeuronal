import pandas as pd
import os
import random

# --- Configuración de Rutas ---
# Asume que este script está en la carpeta 'src/'
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
except NameError:
    # Fallback si se ejecuta en un notebook o REPL interactivo
    data_dir = os.path.join(os.getcwd(), 'data')

# Nombres de archivos
original_csv = os.path.join(data_dir, 'dataset.csv')
aumentado_csv = os.path.join(data_dir, 'dataset_aumentado.csv')

original_jsonl = os.path.join(data_dir, 'dataset.jsonl')
aumentado_jsonl = os.path.join(data_dir, 'dataset_aumentado.jsonl')

print(f"Usando directorio de datos: {data_dir}")

# --- 1. Proceso para CSV ---
print("\n--- Procesando archivos CSV ---")
try:
    if not os.path.exists(aumentado_csv):
        print(f"No se encontró '{aumentado_csv}'. Omitiendo fusión de CSV.")
    elif not os.path.exists(original_csv):
        print(f"No se encontró '{original_csv}'. Renombrando aumentado...")
        os.rename(aumentado_csv, original_csv)
    else:
        df_original = pd.read_csv(original_csv)
        print(f"Datos originales: {len(df_original)} filas.")
        
        df_aumentado = pd.read_csv(aumentado_csv)
        print(f"Datos aumentados: {len(df_aumentado)} filas.")

        # Juntar (concatenar) los dos DataFrames
        df_final = pd.concat([df_original, df_aumentado], ignore_index=True)
        
        # ¡Importante! Mezclar (barajar) el dataset
        # Esto es crucial para que el entrenamiento no vea los datos en orden
        df_final = df_final.sample(frac=1).reset_index(drop=True)
        
        print(f"Datos totales (mezclados): {len(df_final)} filas.")

        # Sobrescribir el 'dataset.csv' original
        df_final.to_csv(original_csv, index=False, encoding='utf-8')
        print(f"✅ Éxito: '{original_csv}' ha sido actualizado.")
        
        # Limpiar el archivo aumentado
        os.remove(aumentado_csv)
        print(f"🧹 Limpieza: '{aumentado_csv}' ha sido eliminado.")

except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo. {e}")
except Exception as e:
    print(f"Ocurrió un error inesperado con el CSV: {e}")

# --- 2. Proceso para JSONL ---
print("\n--- Procesando archivos JSONL ---")
try:
    if not os.path.exists(aumentado_jsonl):
        print(f"No se encontró '{aumentado_jsonl}'. Omitiendo fusión de JSONL.")
    elif not os.path.exists(original_jsonl):
        print(f"No se encontró '{original_jsonl}'. Renombrando aumentado...")
        os.rename(aumentado_jsonl, original_jsonl)
    else:
        all_lines = []
        
        # Leer original
        with open(original_jsonl, 'r', encoding='utf-8') as f:
            lines_original = f.readlines()
            all_lines.extend(lines_original)
        print(f"Datos originales: {len(lines_original)} líneas.")

        # Leer aumentado
        with open(aumentado_jsonl, 'r', encoding='utf-8') as f:
            lines_aumentado = f.readlines()
            all_lines.extend(lines_aumentado)
        print(f"Datos aumentados: {len(lines_aumentado)} líneas.")

        # Mezclar (barajar) las líneas
        random.shuffle(all_lines)
        print(f"Datos totales (mezclados): {len(all_lines)} líneas.")

        # Sobrescribir el 'dataset.jsonl' original
        with open(original_jsonl, 'w', encoding='utf-8') as f:
            f.writelines(all_lines)
        print(f"✅ Éxito: '{original_jsonl}' ha sido actualizado.")

        # Limpiar el archivo aumentado
        os.remove(aumentado_jsonl)
        print(f"🧹 Limpieza: '{aumentado_jsonl}' ha sido eliminado.")
        
except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo. {e}")
except Exception as e:
    print(f"Ocurrió un error inesperado con el JSONL: {e}")

print("\n¡Fusión completada!")