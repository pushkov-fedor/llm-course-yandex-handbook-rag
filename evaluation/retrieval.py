# evaluation/retrieval.py
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
from sentence_transformers import SentenceTransformer

import config
from utils.load_chunks import load_chunks

embedder = SentenceTransformer(config.EMBEDDER_MODEL, device="cpu")
index = faiss.read_index(config.INDEX_FILE)
chunks = load_chunks(os.path.join(os.path.dirname(__file__), "../chunks.jsonl"))

def read_eval_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

questions_items = read_eval_dataset(os.path.join(os.path.dirname(__file__), "../eval_dataset.jsonl"))
questions_items = [questions_item for questions_item in questions_items if questions_item["should_answer"]]

def mrr_k(item: dict, k: int) -> float:
    question = item["question"]
    relevant_docs = item["relevant_docs"]
    relevant_doc_ids = set(doc.replace(".md", "") for doc in relevant_docs)

    embedding = embedder.encode([f"query: {question}"], normalize_embeddings=True)
    _, indices = index.search(embedding, k=k)

    top_k_chunks = [chunks[i] for i in indices[0]] 
    top_k_doc_ids_retrieved = [chunk["doc_id"] for chunk in top_k_chunks]
    print("Found ids: ", top_k_doc_ids_retrieved)
    
    for i, doc_id in enumerate(top_k_doc_ids_retrieved):
        if doc_id in relevant_doc_ids:
            return 1 / (i + 1)
    return 0

def recall_k(item: dict, k: int) -> float:
    question = item["question"]
    relevant_docs = item["relevant_docs"]
    relevant_doc_ids = set(doc.replace(".md", "") for doc in relevant_docs)

    embedding = embedder.encode([f"query: {question}"], normalize_embeddings=True)
    similarities, indices = index.search(embedding, k=k)

    top_k_chunks = [chunks[i] for i in indices[0]] 
    top_k_doc_ids_retrieved = [chunk["doc_id"] for chunk in top_k_chunks]
    print("Found ids: ", top_k_doc_ids_retrieved)
    print("Similarities: ", similarities[0])

    recall = relevant_doc_ids.intersection(top_k_doc_ids_retrieved)
    return len(recall) / len(relevant_doc_ids)

recall_at_k = []
mrr_at_k = []
k = 10

for item in questions_items:
    print("-" * 100)
    print(item["question"])
    print(item["relevant_docs"])
    recall = recall_k(item, k)
    print(f"Recall@{k}: {recall}")
    recall_at_k.append(recall)
    
print("-" * 100)
print(f"Recall@{k}: {recall_at_k}")
print(f"Mean Recall@{k}: {sum(recall_at_k) / len(recall_at_k)}")

print('\n\n\n\n')

for item in questions_items:
    print("-" * 100)
    print(item["question"])
    print(item["relevant_docs"])
    mrr = mrr_k(item, k)
    print(f"MRR@{k}: {mrr}")
    mrr_at_k.append(mrr)
    
print("-" * 100)
print(f"MRR@{k}: {mrr_at_k}")
print(f"Mean MRR@{k}: {sum(mrr_at_k) / len(mrr_at_k)}")


