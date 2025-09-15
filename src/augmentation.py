import pandas as pd
import random
import nltk
from nltk.corpus import wordnet

# === CONFIG ===
FILE = "chat_user_turns_v4_5000-2.csv"
TARGET_CLASS = "Potencial cliente"
AUGMENT_MULTIPLIER = 3  # cuántos ejemplos nuevos por cada original

# === CARGAR DATASET ===
df = pd.read_csv(FILE)

# === FILTRAR SOLO LA CLASE MINORITARIA ===
minority_df = df[df["label"] == TARGET_CLASS]

# === FUNCIONES DE DATA AUGMENTATION ===
def get_synonyms(word):
    """Obtiene sinónimos de una palabra usando WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word, lang='spa'):  # usamos español
        for lemma in syn.lemmas('spa'):
            synonym = lemma.name().replace("_", " ")
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(text, n=1):
    """Reemplaza hasta n palabras por sinónimos"""
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)

    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if synonyms:
            synonym = random.choice(synonyms)
            new_words = [synonym if w == random_word else w for w in new_words]
            num_replaced += 1
        if num_replaced >= n:  # límite de reemplazos
            break

    return " ".join(new_words)

# === AUMENTAR DATA ===
augmented_texts = []
for _, row in minority_df.iterrows():
    for _ in range(AUGMENT_MULTIPLIER):
        new_text = synonym_replacement(row["window_text"], n=2)
        augmented_texts.append({
            "id": row["id"],
            "chat_id": row["chat_id"],
            "msg_idx": row["msg_idx"],
            "label": row["label"],
            "window_text": new_text
        })

aug_df = pd.DataFrame(augmented_texts)

# === CONCATENAR Y GUARDAR ===
df_augmented = pd.concat([df, aug_df], ignore_index=True)
df_augmented.to_csv("chat_user_turns_v4_5000_aug_wordnet.csv", index=False)

print(f"✅ Dataset aumentado guardado: {df_augmented.shape[0]} filas")
print(df_augmented['label'].value_counts())
