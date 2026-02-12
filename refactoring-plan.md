# План рефакторинга

## 1. VectorStore — объединить chunks + embedder + faiss_index

**Файл:** `core/vector_store.py` (заменяет `core/chunks.py`, `core/embedder.py`, `core/faiss_index.py`)

Единый класс, который:
- Загружает чанки из директории
- Загружает FAISS-индекс
- Загружает embedding-модель
- Проверяет инвариант `len(chunks) == index.ntotal` при создании
- Предоставляет метод `search(query, k) -> list[Chunk]`

Зачем: убрать module-level side effects, защитить связку chunks↔index, сделать явный lifecycle.

## 2. Judge — связать промпт + формат ответа + парсинг

**Файл:** `core/judge.py`

Класс, который принимает system_prompt, user_template и response_model (Pydantic), предоставляет метод `evaluate(**kwargs) -> BaseModel`.

Зачем: убрать boilerplate из evaluation-файлов, связать промпт с его response format в одном месте.

## 3. RAGPipeline — retrieve + generate

**Файл:** `core/pipeline.py`

Класс, который оркестрирует Retriever + LLM: `answer(question) -> str`.

Зачем: убрать дублирование цепочки retrieve→format→generate из evaluation-файлов.

## 4. EvaluationRunner — общий run()

**Файл:** `evaluation/runner.py`

Вынести общую обвязку ThreadPoolExecutor + tqdm + сбор результатов из `answer.py`, `retrieval.py`, `refusal.py`.

## 5. Убрать module-level side effects

Все тяжёлые объекты (модель, индекс, чанки) создаются **явно** через конструкторы, а не при импорте модуля. Передаются через DI.

## 6. pyproject.toml

Добавить `pyproject.toml` с `pip install -e .`, чтобы убрать все `sys.path.insert` хаки из evaluation-файлов.

## 7. Мелочи

- Удалить дубликат `HYDE_SYSTEM_PROMPT` из `config.py` (уже есть в `prompts/hyDE.py`)
- Привести `rag_me.py` в рабочее состояние или удалить
- Рассмотреть генерацию JSON Schema из Pydantic (`model_json_schema()`) вместо ручного дублирования в `judge_response.py`

## Целевая файловая структура

```
llm-course/
├── pyproject.toml
├── config.py
├── core/
│   ├── vector_store.py      # chunks + embedder + faiss_index
│   ├── retriever.py          # стратегии retrieval (зависит от VectorStore)
│   ├── llm.py                # обёртка над OpenAI
│   ├── judge.py              # промпт + формат + парсинг
│   └── pipeline.py           # RAGPipeline (retriever + llm)
├── models/
│   ├── chunk.py
│   ├── dataset_item.py
│   └── judge_response.py
├── prompts/
│   ├── hyDE.py
│   ├── judge.py
│   └── rag.py
├── evaluation/
│   ├── runner.py             # общий ThreadPoolExecutor + tqdm
│   ├── answer.py             # correctness + groundedness
│   ├── retrieval.py          # recall + mrr
│   ├── refusal.py
│   └── validate_retrievers.py
└── utils/
    ├── dataset.py
    └── messages.py
```

## Граф зависимостей

```
config
  ↓
core/llm          core/vector_store
  ↓                      ↓
core/judge        core/retriever
      ↓              ↓
     core/pipeline
            ↓
    evaluation/runner
            ↓
 evaluation/{answer,retrieval,refusal}
            ↓
 evaluation/validate_retrievers
```

Зависимости идут строго сверху вниз. `core/` не импортирует из `evaluation/`.

