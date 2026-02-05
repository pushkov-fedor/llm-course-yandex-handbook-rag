import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
from mistralai import Mistral
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import config
from utils.load_chunks import load_chunks

embedder = SentenceTransformer(config.EMBEDDER_MODEL, device="cpu")
index = faiss.read_index(config.INDEX_FILE)
chunks = load_chunks(os.path.join(os.path.dirname(__file__), "../chunks"))

llm_client = Mistral(api_key=config.MISTRAL_API_KEY)


class CorrectnessJudgment(BaseModel):
    score: int
    explanation: str


JUDGE_SYSTEM_PROMPT = """Ты — эксперт по оценке качества ответов на вопросы по машинному обучению.
Твоя задача — оценить ПРАВИЛЬНОСТЬ (correctness) ответа модели.

ВАЖНО: Оценивай фактическую корректность ответа, полноту и точность изложения.
Сравнивай с эталонным ответом, но учитывай, что модель может дать правильный ответ другими словами."""

JUDGE_USER_PROMPT = """Вопрос: {question}

Эталонный ответ (ground truth):
{reference_answer}

---

Ответ модели:
{model_answer}

---

Оцени правильность (correctness) ответа модели по шкале от 1 до 5:

5 — Отлично: Ответ полностью правильный, содержит все ключевые факты из эталона, точен.
4 — Хорошо: Ответ в целом правильный, охватывает основные моменты, возможны незначительные упущения.
3 — Удовлетворительно: Ответ частично правильный, но упущены важные детали или есть неточности.
2 — Плохо: Ответ содержит существенные ошибки или упускает большую часть важной информации.
1 — Очень плохо: Ответ неправильный, содержит грубые фактические ошибки или не отвечает на вопрос."""


def read_eval_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def retrieve_context(question: str, k: int = 10) -> tuple[list[dict], str]:
    embedding = embedder.encode([question], normalize_embeddings=True)
    similarities, indices = index.search(embedding, k=k)
    
    top_k_chunks = [chunks[i] for i in indices[0]]
    
    context_parts = []
    for i, chunk in enumerate(top_k_chunks, 1):
        source = f"Источник {i}: {chunk['article_num']}. {chunk['article_title']}"
        context_parts.append(f"--- {source} ---\n{chunk['text']}")
    
    formatted_context = "\n\n".join(context_parts)
    
    return top_k_chunks, formatted_context


def generate_answer(question: str, context: str) -> str:
    user_message = config.RAG_USER_TEMPLATE.format(question=question, context=context)

    response = llm_client.chat.complete(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": config.RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1024
    )
    
    return response.choices[0].message.content


def evaluate_correctness(question: str, reference_answer: str, model_answer: str) -> dict:
    judge_prompt = JUDGE_USER_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        model_answer=model_answer
    )
    
    try:
        response = llm_client.chat.parse(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0.0,
            response_format=CorrectnessJudgment
        )
        
        judgment = response.choices[0].message.parsed
        
        return {
            "score": judgment.score,
            "explanation": judgment.explanation
        }
    except Exception as e:
        print(f"⚠️  Ошибка при оценке correctness: {e}")
        return {
            "score": 0,
            "explanation": f"Ошибка оценки: {str(e)}"
        }


def evaluate_dataset(dataset_path: str, k: int = 10, limit: int = None) -> list[dict]:
    dataset = read_eval_dataset(dataset_path)
    
    dataset = [item for item in dataset if item["should_answer"]]
    
    if limit:
        dataset = dataset[:limit]
    
    results = []
    
    print(f"🔍 Начинаю оценку correctness на {len(dataset)} примерах (k={k})...\n")
    print("=" * 100)
    
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        reference_answer = item["answer"]
        
        print(f"\n[{i}/{len(dataset)}] Вопрос: {question}")
        
        retrieved_chunks, context = retrieve_context(question, k=k)
        retrieved_doc_ids = [chunk["doc_id"] for chunk in retrieved_chunks]
        print(f"📄 Найдены документы: {retrieved_doc_ids[:5]}...")
        
        print("⏳ Генерирую ответ...")
        answer = generate_answer(question, context)
        print(f"💬 Ответ: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        
        print("⚖️  Оцениваю correctness...")
        evaluation = evaluate_correctness(question, reference_answer, answer)
        
        print(f"📊 Correctness: {evaluation['score']}/5")
        print(f"📝 Объяснение: {evaluation['explanation']}")
        print("=" * 100)
        
        results.append({
            "id": item["id"],
            "question": question,
            "reference_answer": reference_answer,
            "model_answer": answer,
            "context_doc_ids": retrieved_doc_ids,
            "relevant_doc_ids": item["relevant_docs"],
            "correctness_score": evaluation["score"],
            "correctness_explanation": evaluation["explanation"]
        })
        
        if i < len(dataset):
            print("\n⏸️  Пауза 15 секунд перед следующим примером...")
            time.sleep(15)
    
    return results


def compute_metrics(results: list[dict]) -> dict:
    scores = [r["correctness_score"] for r in results]
    
    normalized_scores = [(s - 1) / 4 for s in scores]
    
    return {
        "mean_correctness": sum(scores) / len(scores),
        "mean_correctness_normalized": sum(normalized_scores) / len(normalized_scores),
        "min_correctness": min(scores),
        "max_correctness": max(scores),
        "perfect_score_count": sum(1 for s in scores if s == 5),
        "poor_score_count": sum(1 for s in scores if s <= 2),
        "total_evaluated": len(results)
    }


def save_results(results: list[dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\n💾 Результаты сохранены в {output_path}")


def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "../eval_dataset.jsonl")
    output_path = os.path.join(os.path.dirname(__file__), "correctness_results.jsonl")
    
    results = evaluate_dataset(dataset_path, k=10, limit=None)
    
    metrics = compute_metrics(results)
    
    print("\n" + "=" * 100)
    print("📊 ИТОГОВЫЕ МЕТРИКИ CORRECTNESS")
    print("=" * 100)
    print(f"Средний балл (1-5):        {metrics['mean_correctness']:.2f}")
    print(f"Средний балл (0-1):        {metrics['mean_correctness_normalized']:.2f}")
    print(f"Минимальный балл:          {metrics['min_correctness']:.0f}")
    print(f"Максимальный балл:         {metrics['max_correctness']:.0f}")
    print(f"Идеальных ответов (5/5):   {metrics['perfect_score_count']}")
    print(f"Плохих ответов (≤2/5):     {metrics['poor_score_count']}")
    print(f"Всего оценено:             {metrics['total_evaluated']}")
    print("=" * 100)
    
    save_results(results, output_path)
    
    metrics_path = os.path.join(os.path.dirname(__file__), "correctness_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"📈 Метрики сохранены в {metrics_path}")


if __name__ == "__main__":
    main()

