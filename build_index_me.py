import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from utils.load_chunks import load_chunks

embedder = SentenceTransformer(config.EMBEDDER_MODEL, device="cpu")

chunks = load_chunks("chunks")

texts = [chunk['text'] for chunk in chunks]

embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings.astype(np.float32))

faiss.write_index(index, config.INDEX_FILE)
print(f"Index saved to {config.INDEX_FILE}")