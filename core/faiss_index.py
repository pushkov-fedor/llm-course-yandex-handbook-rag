import faiss
import numpy as np

import config

index = faiss.read_index(config.INDEX_FILE)

def search_in_index(embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    return index.search(embedding, k=k)