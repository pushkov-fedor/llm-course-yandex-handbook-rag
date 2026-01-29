#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /home/fedor/Study/llm-course/build_index.py
"""
Скрипт для создания FAISS индекса из чанков.
Вычисляет embeddings для всех чанков и сохраняет индекс.
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Конфигурация
CHUNKS_FILE = "chunks.jsonl"
INDEX_FILE = "faiss_index.bin"
CHUNKS_META_FILE = "chunks_meta.pkl"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"  # Хорошо работает с русским
BATCH_SIZE = 32


def load_chunks(path: str) -> list[dict]:
    """Загружает чанки из jsonl файла."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_index():
    """Создаёт FAISS индекс из чанков."""
    print(f"Загрузка модели {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    
    print(f"Загрузка чанков из {CHUNKS_FILE}...")
    chunks = load_chunks(CHUNKS_FILE)
    print(f"Загружено {len(chunks)} чанков")
    
    # Подготовка текстов для E5 модели (требует префикс "passage: ")
    texts = [f"passage: {chunk['text']}" for chunk in chunks]
    
    print("Вычисление embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Для косинусного сходства через inner product
    )
    
    # Создание FAISS индекса (Inner Product для нормализованных векторов = косинусное сходство)
    dimension = embeddings.shape[1]
    print(f"Размерность embeddings: {dimension}")
    
    index = faiss.IndexFlatIP(dimension)  # Inner Product
    index.add(embeddings.astype(np.float32))
    
    print(f"Индекс создан, всего векторов: {index.ntotal}")
    
    # Сохранение индекса
    faiss.write_index(index, INDEX_FILE)
    print(f"Индекс сохранён в {INDEX_FILE}")
    
    # Сохранение метаданных чанков
    with open(CHUNKS_META_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Метаданные сохранены в {CHUNKS_META_FILE}")
    
    print("\n✅ Готово!")
    print(f"   - Индекс: {INDEX_FILE} ({Path(INDEX_FILE).stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"   - Метаданные: {CHUNKS_META_FILE}")


if __name__ == "__main__":
    build_index()

