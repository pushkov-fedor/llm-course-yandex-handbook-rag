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
chunks = load_chunks(os.path.join(os.path.dirname(__file__), "../chunks"))

def read_eval_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def mrr_k(item: dict, k: int, verbose: bool = True) -> float:
    question = item["question"]
    relevant_docs = item["relevant_docs"]
    relevant_doc_ids = set(doc.replace(".md", "") for doc in relevant_docs)

    embedding = embedder.encode([question], normalize_embeddings=True)
    _, indices = index.search(embedding, k=k)

    top_k_chunks = [chunks[i] for i in indices[0]] 
    top_k_doc_ids_retrieved = [chunk["doc_id"] for chunk in top_k_chunks]
    
    if verbose:
        print(f"📄 Найдены документы: {top_k_doc_ids_retrieved[:5]}{'...' if len(top_k_doc_ids_retrieved) > 5 else ''}")
        print(f"🎯 Релевантные документы: {list(relevant_doc_ids)}")
    
    for i, doc_id in enumerate(top_k_doc_ids_retrieved):
        if doc_id in relevant_doc_ids:
            mrr = 1 / (i + 1)
            if verbose:
                print(f"✓ Первый релевантный на позиции: {i+1}")
            return mrr
    
    if verbose:
        print(f"✗ Релевантные документы не найдены в топ-{k}")
    return 0.0

def mrr_k_chunks(item: dict, k: int, verbose: bool = True) -> float:
    question = item["question"]
    relevant_chunk_ids = set(item["relevant_chunk_ids"])

    embedding = embedder.encode([question], normalize_embeddings=True)
    _, indices = index.search(embedding, k=k)

    top_k_chunks = [chunks[i] for i in indices[0]] 
    top_k_chunk_ids_retrieved = [chunk["chunk_id"] for chunk in top_k_chunks]
    
    if verbose:
        print(f"📄 Найдены чанки: {top_k_chunk_ids_retrieved[:3]}{'...' if len(top_k_chunk_ids_retrieved) > 3 else ''}")
        print(f"🎯 Релевантные чанки: {list(relevant_chunk_ids)[:3]}{'...' if len(relevant_chunk_ids) > 3 else ''}")
    
    for i, chunk_id in enumerate(top_k_chunk_ids_retrieved):
        if chunk_id in relevant_chunk_ids:
            mrr = 1 / (i + 1)
            if verbose:
                print(f"✓ Первый релевантный чанк на позиции: {i+1}")
            return mrr
    
    if verbose:
        print(f"✗ Релевантные чанки не найдены в топ-{k}")
    return 0.0

def evaluate_mrr_at_k(questions_items: list[dict], k: int) -> dict:
    mrr_scores = []
    
    print(f"🔍 Начинаю оценку MRR@{k} на {len(questions_items)} примерах...\n")
    print("=" * 100)
    
    for i, item in enumerate(questions_items, 1):
        print(f"\n[{i}/{len(questions_items)}] Вопрос: {item['question']}")
        
        mrr = mrr_k(item, k, verbose=True)
        print(f"📊 MRR@{k}: {mrr:.3f}")
        mrr_scores.append(mrr)
        
        if i < len(questions_items):
            print("-" * 100)
    
    print("\n" + "=" * 100)
    
    return {
        "metric": f"MRR@{k}",
        "k": k,
        "scores": mrr_scores,
        "mean": sum(mrr_scores) / len(mrr_scores),
        "min": min(mrr_scores),
        "max": max(mrr_scores),
        "perfect_count": sum(1 for s in mrr_scores if s == 1.0),
        "zero_count": sum(1 for s in mrr_scores if s == 0.0),
        "total": len(mrr_scores)
    }

def print_metrics(results: dict):
    print(f"📊 ИТОГОВЫЕ МЕТРИКИ: {results['metric']}")
    print("=" * 100)
    print(f"Средний MRR:               {results['mean']:.3f}")
    print(f"Минимальный MRR:           {results['min']:.3f}")
    print(f"Максимальный MRR:          {results['max']:.3f}")
    print(f"Идеальных результатов (1.0): {results['perfect_count']}")
    print(f"Нулевых результатов (0.0):   {results['zero_count']}")
    print(f"Всего оценено:               {results['total']}")
    print("=" * 100)

def save_results(results: dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Результаты сохранены в {output_path}")

def evaluate_mrr_at_k_chunks(questions_items: list[dict], k: int) -> dict:
    mrr_scores = []
    
    print(f"🔍 Начинаю оценку MRR@{k} по чанкам на {len(questions_items)} примерах...\n")
    print("=" * 100)
    
    for i, item in enumerate(questions_items, 1):
        print(f"\n[{i}/{len(questions_items)}] Вопрос: {item['question']}")
        
        mrr = mrr_k_chunks(item, k, verbose=True)
        print(f"📊 MRR@{k}: {mrr:.3f}")
        mrr_scores.append(mrr)
        
        if i < len(questions_items):
            print("-" * 100)
    
    print("\n" + "=" * 100)
    
    return {
        "metric": f"MRR@{k} (chunks)",
        "k": k,
        "scores": mrr_scores,
        "mean": sum(mrr_scores) / len(mrr_scores),
        "min": min(mrr_scores),
        "max": max(mrr_scores),
        "perfect_count": sum(1 for s in mrr_scores if s == 1.0),
        "zero_count": sum(1 for s in mrr_scores if s == 0.0),
        "total": len(mrr_scores)
    }

if __name__ == "__main__":
    questions_items = read_eval_dataset(os.path.join(os.path.dirname(__file__), "../eval_dataset.jsonl"))
    questions_items = [questions_item for questions_item in questions_items if questions_item["should_answer"]]
    
    k = 10
    
    print("\n" + "🏆" * 50)
    print("ОЦЕНКА MRR@K ПО ДОКУМЕНТАМ")
    print("🏆" * 50 + "\n")
    results_docs = evaluate_mrr_at_k(questions_items, k)
    print_metrics(results_docs)
    save_results(results_docs, os.path.join(os.path.dirname(__file__), "mrr_docs_metrics.json"))
    
    print("\n\n" + "📦" * 50)
    print("ОЦЕНКА MRR@K ПО ЧАНКАМ")
    print("📦" * 50 + "\n")
    results_chunks = evaluate_mrr_at_k_chunks(questions_items, k)
    print_metrics(results_chunks)
    save_results(results_chunks, os.path.join(os.path.dirname(__file__), "mrr_chunks_metrics.json"))
    
    print("\n✅ Оценка завершена!")

