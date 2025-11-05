import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

INPUT_CSV = "data/cleaned_pages.csv"
OUTPUT_JSONL = "data/embedded_pages.jsonl"
MODEL_NAME = "intfloat/e5-base-v2"

def load_model():
    print(f"🔹 Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    return model

def chunk_text(text, max_length=512):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield " ".join(words[i:i+max_length])

def embed_texts():
    model = load_model()
    df = pd.read_csv(INPUT_CSV)
    
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            text = row["cleaned_content"]
            if not isinstance(text, str) or not text.strip():
                continue

            for chunk in chunk_text(text):
                emb = model.encode(chunk, normalize_embeddings=True)
                record = {
                    "url": row["url"],
                    "title": row["title"],
                    "text": chunk,
                    "embedding": emb.tolist()
                }
                f.write(json.dumps(record) + "\n")
    print(f"Embeddings saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    embed_texts()
