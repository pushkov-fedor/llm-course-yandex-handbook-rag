import numpy as np
from sentence_transformers import SentenceTransformer

import config

embedder = SentenceTransformer(config.EMBEDDER_MODEL, device="cpu")

def encode(text: str) -> np.ndarray:
    return embedder.encode([text], normalize_embeddings=True)