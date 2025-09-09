import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import re

# ----------------------------
# 1️⃣ Cargar dataset
# ----------------------------
df = pd.read_csv(r"C:\RedNeuronal\chat_user_turns_v4_5000_aug_wordnet.csv")  # Ajusta con tu path

# ----------------------------
# 2️⃣ Limpiar datos
# ----------------------------
# Eliminar duplicados exactos
df = df.drop_duplicates(subset=['window_text'])

# Filtrar textos demasiado cortos
df = df[df['window_text'].str.len() > 10]

# Normalizar texto: minusculas y eliminar caracteres extra
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-záéíóúñü0-9\s<>]", "", text)  # Mantener <usr> y <sys>
    return text

df['window_text'] = df['window_text'].apply(clean_text)

# Guardar dataset limpio opcionalmente
df.to_csv(r"C:\RedNeuronal\chat_user_turns_v4_clean.csv", index=False)
print("✅ Dataset limpio guardado:", len(df), "filas")

# ----------------------------
# 3️⃣ Label Encoding
# ----------------------------
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])
num_classes = len(le.classes_)

# ----------------------------
# 4️⃣ Tokenización y vocabulario
# ----------------------------
all_text = " ".join(df['window_text'].tolist()).split()
word_counts = Counter(all_text)
vocab = {word: idx+1 for idx, (word, _) in enumerate(word_counts.most_common())}  # 0 = padding
vocab_size = len(vocab) + 1

max_len = 250
def tokenize(text):
    tokens = [vocab.get(word, 0) for word in text.split()]
    if len(tokens) < max_len:
        tokens = tokens + [0]*(max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens

df['tokens'] = df['window_text'].apply(tokenize)

# ----------------------------
# 5️⃣ Dataset y DataLoader PyTorch
# ----------------------------
class ChatDataset(Dataset):
    def __init__(self, df):
        self.texts = df['tokens'].tolist()
        self.labels = df['label_enc'].tolist()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.texts[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

dataset = ChatDataset(df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ----------------------------
# 6️⃣ Comprobación
# ----------------------------
for x_batch, y_batch in dataloader:
    print("Batch de tokens:", x_batch.shape)
    print("Batch de labels:", y_batch.shape)
    break
