# /home/fedor/Study/llm-course/rag_me.py
import json
import os

import faiss
from mistralai import Mistral
from sentence_transformers import SentenceTransformer

import config
from utils.load_chunks import load_chunks

llm = Mistral(api_key=config.MISTRAL_API_KEY)
embedder = SentenceTransformer(config.EMBEDDER_MODEL, device="cpu")
index = faiss.read_index(config.INDEX_FILE)

chunks = load_chunks(os.path.join(os.path.dirname(__file__), "../chunks.jsonl"))

prompt = """
Ты — помощник по учебнику машинного обучения от Яндекса.
Отвечай на вопросы ТОЛЬКО на основе предоставленного контекста из учебника.
Если в контексте нет информации для ответа — честно скажи об этом.
Отвечай на русском языке. Будь точен и лаконичен.
При необходимости используй LaTeX для формул (в формате $..$ для inline и $$...$$ для block).
"""
def answer_question(question: str) -> str:
    embedding = embedder.encode([f"query: {question}"], normalize_embeddings=True)
    similarities, indices = index.search(embedding, k=5)

    print('Similarities: ', similarities[0])

    top_k_chunks = [chunks[i] for i in indices[0]]

    question_with_context = """
Вопрос пользователя: {question}

Вот что мы нашли в учебнике:

{context}

Ответь на вопрос пользователя на основе предоставленного контекста. 
В тексте ссылайся на источник в формате: (Источник: номер и название статьи).
НЕ добавляй URL-ссылки — только номер и название статьи.
"""

    context = [f"--- Источник: {chunk['article_num']}. {chunk['article_title']} ---\n{chunk['text']}" for chunk in top_k_chunks]
    user_request = question_with_context.format(question=question, context='\n\n'.join(context))

    print(user_request + '\n\n' + '-' * 100 + '\n\n')

    response = llm.chat.complete(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_request},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("=" * 60)
    print("RAG-помощник по учебнику ML от Яндекса")
    print("=" * 60)
    
    while True:
        question = input("\n🔍 Введите вопрос (или 'q' для выхода): ").strip()
        if question.lower() in ('q', 'quit', 'exit', ''):
            print("До свидания!")
            break
        print("\n" + "-" * 60 + "\n")
        print(answer_question(question))