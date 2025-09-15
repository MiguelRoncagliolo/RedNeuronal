import pandas as pd

# === CONFIG ===
FILE = "../data/dataset.csv"  # Ajusta si usas otro archivo

# === CARGAR DATASET ===
df = pd.read_csv(FILE)

print("📌 Primeras filas del dataset:")
print(df.head(), "\n")

# === DISTRIBUCIÓN DE CLASES ===
print("📊 Distribución de clases:")
print(df["label"].value_counts(), "\n")

# === DUPLICADOS ===
duplicados = df.duplicated().sum()
print(f"🔎 Número de filas duplicadas: {duplicados}\n")

# === ESTADÍSTICAS DEL TEXTO ===
df["text_len"] = df["window_text"].astype(str).apply(len)

print("📏 Estadísticas del largo de los textos:")
print(df["text_len"].describe(), "\n")

print("Ejemplo de texto más corto:", df.loc[df["text_len"].idxmin(), "window_text"])
print("Ejemplo de texto más largo:", df.loc[df["text_len"].idxmax(), "window_text"])
