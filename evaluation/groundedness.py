# evaluation/groundedness.py
"""
Оценка groundedness (обоснованности) ответов RAG-системы.
Проверяет, насколько ответ основан на предоставленном контексте.
"""
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

# Инициализация компонентов
embedder = SentenceTransformer(config.EMBEDDER_MODEL, device="cpu")
index = faiss.read_index(config.INDEX_FILE)
chunks = load_chunks(os.path.join(os.path.dirname(__file__), "../chunks"))

llm_client = Mistral(api_key=config.MISTRAL_API_KEY)


class GroundednessJudgment(BaseModel):
    score: int
    explanation: str


# Промпт для оценки groundedness
JUDGE_SYSTEM_PROMPT = """Ты — строгий эксперт по оценке качества ответов RAG-систем.
Твоя задача — оценить, насколько ответ модели ОБОСНОВАН предоставленным контекстом.

ВАЖНО: Ответ должен содержать ТОЛЬКО информацию из контекста. Любые факты, определения, 
утверждения, которых НЕТ в контексте — это галлюцинация."""

JUDGE_USER_PROMPT = """Контекст из учебника:

{context}

---

Вопрос пользователя: {question}

Ответ модели: {answer}

---

Оцени обоснованность (groundedness) ответа по шкале от 1 до 5:

5 — Отлично: Каждое утверждение напрямую подтверждается контекстом. Никаких галлюцинаций.
4 — Хорошо: Ответ в основном основан на контексте, незначительные переформулировки.
3 — Удовлетворительно: Основные факты из контекста, но есть информация, которую сложно проверить.
2 — Плохо: Значительная часть не подтверждается контекстом или содержит сомнительные утверждения.
1 — Очень плохо: Явные галлюцинации или факты, которых нет в контексте."""


def read_eval_dataset(path: str) -> list[dict]:
    """Читает датасет для оценки."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def retrieve_context(question: str, k: int = 5) -> tuple[list[dict], str]:
    """
    Получает релевантные чанки для вопроса.
    
    Returns:
        (list[dict], str): Список чанков и отформатированный контекст
    """
    embedding = embedder.encode([question], normalize_embeddings=True)
    similarities, indices = index.search(embedding, k=k)
    
    top_k_chunks = [chunks[i] for i in indices[0]]
    
    # Форматируем контекст для LLM
    context_parts = []
    for i, chunk in enumerate(top_k_chunks, 1):
        source = f"Источник {i}: {chunk['article_num']}. {chunk['article_title']}"
        context_parts.append(f"--- {source} ---\n{chunk['text']}")
    
    formatted_context = "\n\n".join(context_parts)
    
    return top_k_chunks, formatted_context


def generate_answer(question: str, context: str) -> str:
    """Генерирует ответ с помощью RAG-модели."""
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


def evaluate_groundedness(question: str, context: str, answer: str) -> dict:
    """
    Оценивает groundedness ответа с помощью модели-судьи.
    Использует structured output для надежного получения оценки.
    
    Returns:
        dict: {"score": int (1-5), "explanation": str}
    """
    judge_prompt = JUDGE_USER_PROMPT.format(
        context=context,
        question=question,
        answer=answer
    )
    
    try:
        response = llm_client.chat.parse(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0.0,
            response_format=GroundednessJudgment
        )
        
        judgment = response.choices[0].message.parsed
        
        return {
            "score": judgment.score,
            "explanation": judgment.explanation
        }
    except Exception as e:
        print(f"⚠️  Ошибка при оценке groundedness: {e}")
        return {
            "score": 0,
            "explanation": f"Ошибка оценки: {str(e)}"
        }


def evaluate_dataset(dataset_path: str, k: int = 5, limit: int = None) -> list[dict]:
    """
    Оценивает groundedness на датасете.
    
    Args:
        dataset_path: Путь к eval_dataset.jsonl
        k: Количество чанков для retrieval
        limit: Ограничение на количество примеров (для отладки)
    
    Returns:
        list[dict]: Результаты оценки для каждого примера
    """
    dataset = read_eval_dataset(dataset_path)
    
    # Берем только те, на которые должны отвечать
    dataset = [item for item in dataset if item["should_answer"]]
    
    if limit:
        dataset = dataset[:limit]
    
    results = []
    
    print(f"🔍 Начинаю оценку groundedness на {len(dataset)} примерах...\n")
    print("=" * 100)
    
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        
        print(f"\n[{i}/{len(dataset)}] Вопрос: {question}")
        
        # 1. Получаем контекст
        retrieved_chunks, context = retrieve_context(question, k=k)
        retrieved_doc_ids = [chunk["doc_id"] for chunk in retrieved_chunks]
        print(f"📄 Найдены документы: {retrieved_doc_ids}")
        
        # 2. Генерируем ответ
        print("⏳ Генерирую ответ...")
        answer = generate_answer(question, context)
        print(f"💬 Ответ: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        
        # 3. Оцениваем groundedness
        print("⚖️  Оцениваю groundedness...")
        evaluation = evaluate_groundedness(question, context, answer)
        
        print(f"📊 Groundedness: {evaluation['score']}/5")
        print(f"📝 Объяснение: {evaluation['explanation']}")
        print("=" * 100)
        
        results.append({
            "id": item["id"],
            "question": question,
            "answer": answer,
            "context_doc_ids": retrieved_doc_ids,
            "relevant_doc_ids": item["relevant_docs"],
            "groundedness_score": evaluation["score"],
            "groundedness_explanation": evaluation["explanation"]
        })
        
        # Пауза между примерами (кроме последнего)
        if i < len(dataset):
            print("\n⏸️  Пауза 15 секунд перед следующим примером...")
            time.sleep(15)
    
    return results


def compute_metrics(results: list[dict]) -> dict:
    """Вычисляет агрегированные метрики."""
    scores = [r["groundedness_score"] for r in results]
    
    # Нормализуем к [0, 1] для удобства
    normalized_scores = [(s - 1) / 4 for s in scores]
    
    return {
        "mean_groundedness": sum(scores) / len(scores),
        "mean_groundedness_normalized": sum(normalized_scores) / len(normalized_scores),
        "min_groundedness": min(scores),
        "max_groundedness": max(scores),
        "perfect_score_count": sum(1 for s in scores if s == 5),
        "poor_score_count": sum(1 for s in scores if s <= 2),
        "total_evaluated": len(results)
    }


def save_results(results: list[dict], output_path: str):
    """Сохраняет результаты в JSONL."""
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\n💾 Результаты сохранены в {output_path}")


def main():
    """Основная функция для запуска оценки."""
    dataset_path = os.path.join(os.path.dirname(__file__), "../eval_dataset.jsonl")
    output_path = os.path.join(os.path.dirname(__file__), "groundedness_results.jsonl")
    
    # Для отладки можно использовать limit=5
    results = evaluate_dataset(dataset_path, k=5, limit=None)
    
    # Вычисляем метрики
    metrics = compute_metrics(results)
    
    print("\n" + "=" * 100)
    print("📊 ИТОГОВЫЕ МЕТРИКИ GROUNDEDNESS")
    print("=" * 100)
    print(f"Средний балл (1-5):        {metrics['mean_groundedness']:.2f}")
    print(f"Средний балл (0-1):        {metrics['mean_groundedness_normalized']:.2f}")
    print(f"Минимальный балл:          {metrics['min_groundedness']:.0f}")
    print(f"Максимальный балл:         {metrics['max_groundedness']:.0f}")
    print(f"Идеальных ответов (5/5):   {metrics['perfect_score_count']}")
    print(f"Плохих ответов (≤2/5):     {metrics['poor_score_count']}")
    print(f"Всего оценено:             {metrics['total_evaluated']}")
    print("=" * 100)
    
    # Сохраняем результаты
    save_results(results, output_path)
    
    # Сохраняем метрики отдельно
    metrics_path = os.path.join(os.path.dirname(__file__), "groundedness_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"📈 Метрики сохранены в {metrics_path}")


if __name__ == "__main__":
    main()

