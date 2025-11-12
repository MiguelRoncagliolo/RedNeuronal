import pandas as pd

FILE = "../../data/dataset.jsonl"

df = pd.read_json(FILE, lines=True)

print("Primeras filas del dataset:")
print(df.head(), "\n")

print("Distribucion de clases:")
print(df["label"].value_counts(), "\n")

duplicados = df.duplicated().sum()
print(f"Numero de filas duplicadas: {duplicados}\n")

df["text_len"] = df["window_text"].astype(str).apply(len)

print("Estadisticas del largo de los textos:")
print(df["text_len"].describe(), "\n")

print("Ejemplo de texto más corto:", df.loc[df["text_len"].idxmin(), "window_text"])
print("Ejemplo de texto más largo:", df.loc[df["text_len"].idxmax(), "window_text"])
