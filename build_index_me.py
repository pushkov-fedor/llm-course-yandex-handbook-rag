import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config

embedder = SentenceTransformer(config.EMBEDDER_MODEL, device="cpu")

def load_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks

chunks = load_chunks("chunks.jsonl")

texts = [f"passage: {chunk['text']}" for chunk in chunks]

embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings.astype(np.float32))

faiss.write_index(index, config.INDEX_FILE)
print(f"Index saved to {config.INDEX_FILE}")