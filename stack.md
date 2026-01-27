# Технический стек проекта

## Язык и окружение
- Python 3.10+
- Conda

## LLM
- Mistral через API
- Пакет `mistralai`
  - генерация ответов
  - официальный токенайзер (SentencePiece)

## Чанкинг
- Резка текста **по токенам** с использованием токенайзера Mistral
- Контроль размера чанков и контекста в токенах

## Embeddings
- Отдельная embedding-модель (не Mistral)
  - `bge-*` или `multilingual-e5-*`
- Библиотека: `sentence-transformers`

## Поиск
- FAISS (dense retrieval)
- Опционально: BM25 для гибридного поиска

## RAG-логика
- Собственная реализация (router, multi-step retrieval, self-check)
- Без “просто ChatGPT по промпту”

## Интерфейс
- Streamlit (web)
- Опционально: Telegram-бот (aiogram)

## Оценка и качество
- pytest
- pandas / numpy
- Retrieval- и answer-метрики

## Стиль и качество кода
- PEP8
- ruff / black
- GitHub Actions (lint + tests)
