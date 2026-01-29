#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /home/fedor/Study/llm-course/rag.py
"""
RAG CLI для поиска по учебнику ML.
Использует FAISS для retrieval и Mistral для генерации ответов.
"""

import os
import pickle

import faiss
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral
from sentence_transformers import SentenceTransformer

# Загрузка переменных из .env
load_dotenv()

# Конфигурация
INDEX_FILE = "faiss_index.bin"
CHUNKS_META_FILE = "chunks_meta.pkl"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
LLM_MODEL = "mistral-small-latest"
TOP_K = 5

# Системный промпт для RAG
SYSTEM_PROMPT = """Ты — помощник по учебнику машинного обучения от Яндекса.
Отвечай на вопросы ТОЛЬКО на основе предоставленного контекста из учебника.
Если в контексте нет информации для ответа — честно скажи об этом.
Отвечай на русском языке. Будь точен и лаконичен.
При необходимости используй LaTeX для формул (в формате $..$ для inline и $$...$$ для block)."""


class RAGSystem:
    def __init__(self):
        print("Загрузка компонентов RAG...")
        
        # Загрузка embedding модели
        print(f"  - Модель embeddings: {EMBEDDING_MODEL}")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        
        # Загрузка FAISS индекса
        print(f"  - FAISS индекс: {INDEX_FILE}")
        self.index = faiss.read_index(INDEX_FILE)
        
        # Загрузка метаданных чанков
        print(f"  - Метаданные: {CHUNKS_META_FILE}")
        with open(CHUNKS_META_FILE, "rb") as f:
            self.chunks = pickle.load(f)
        
        # Инициализация Mistral клиента
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Установите переменную окружения MISTRAL_API_KEY")
        self.llm_client = Mistral(api_key=api_key)
        
        print(f"✅ RAG готов к работе! ({len(self.chunks)} чанков в индексе)\n")
    
    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Поиск релевантных чанков по запросу."""
        # E5 модель требует префикс "query: " для запросов
        query_text = f"query: {query}"
        
        # Получение embedding запроса
        query_embedding = self.embed_model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Поиск в FAISS
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Формирование результатов
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(score)
                results.append(chunk)
        
        return results
    
    def format_context(self, chunks: list[dict]) -> str:
        """Форматирует чанки в контекст для LLM."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = f"[{chunk['chapter_num']}. {chunk['chapter_title']} → {chunk['article_num']} {chunk['article_title']}]"
            context_parts.append(f"--- Источник {i}: {source} ---\n{chunk['text']}")
        return "\n\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Генерирует ответ с помощью LLM."""
        user_message = f"""Контекст из учебника:

{context}

---

Вопрос: {query}

Ответь на вопрос, используя только информацию из контекста выше."""

        response = self.llm_client.chat.complete(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    
    def ask(self, query: str, top_k: int = TOP_K, show_sources: bool = True) -> str:
        """Полный RAG пайплайн: поиск + генерация ответа."""
        # 1. Поиск релевантных чанков
        chunks = self.search(query, top_k)
        
        if not chunks:
            return "Не найдено релевантных фрагментов в учебнике."
        
        # 2. Формирование контекста
        context = self.format_context(chunks)
        
        # 3. Генерация ответа
        answer = self.generate_answer(query, context)
        
        # 4. Добавление источников
        if show_sources:
            sources = "\n\n📚 **Источники:**"
            for i, chunk in enumerate(chunks, 1):
                sources += f"\n{i}. [{chunk['article_num']} {chunk['article_title']}]({chunk['source_url']}) (score: {chunk['score']:.3f})"
            answer += sources
        
        return answer


def main():
    """CLI интерфейс для RAG."""
    print("=" * 60)
    print("🎓 RAG-поисковик по учебнику ML (Яндекс)")
    print("=" * 60)
    print()
    
    try:
        rag = RAGSystem()
    except ValueError as e:
        print(f"❌ Ошибка: {e}")
        print("\nДля работы нужен API ключ Mistral:")
        print("  export MISTRAL_API_KEY='ваш_ключ'")
        return
    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e}")
        print("\nСначала создайте индекс:")
        print("  python build_index.py")
        return
    
    print("Введите вопрос (или 'exit' для выхода, 'search' для только поиска):")
    print("-" * 60)
    
    while True:
        try:
            query = input("\n❓ Вопрос: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nДо свидания! 👋")
            break
        
        if not query:
            continue
        
        if query.lower() in ("exit", "quit", "q", "выход"):
            print("\nДо свидания! 👋")
            break
        
        # Режим только поиска (без LLM)
        if query.lower().startswith("search "):
            search_query = query[7:].strip()
            if search_query:
                print("\n🔍 Результаты поиска:")
                results = rag.search(search_query, top_k=5)
                for i, chunk in enumerate(results, 1):
                    print(f"\n--- Результат {i} (score: {chunk['score']:.3f}) ---")
                    print(f"📖 {chunk['article_num']} {chunk['article_title']}")
                    print(f"🔗 {chunk['source_url']}")
                    print(f"\n{chunk['text'][:500]}...")
            continue
        
        # Полный RAG пайплайн
        print("\n⏳ Ищу ответ...")
        try:
            answer = rag.ask(query)
            print(f"\n💡 **Ответ:**\n\n{answer}")
        except Exception as e:
            print(f"\n❌ Ошибка при генерации ответа: {e}")


if __name__ == "__main__":
    main()

