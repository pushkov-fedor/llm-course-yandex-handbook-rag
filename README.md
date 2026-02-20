# 🔍 Поисковик по учебнику Яндекса по ML

RAG-система для поиска по [учебнику Яндекса по ML](https://education.yandex.ru/handbook/ml), реализованная в рамках учебного курса.

**Модель:** Mistral Small 3.1 24B &nbsp;|&nbsp; **Провайдер:** [OpenRouter](https://openrouter.ai) &nbsp;|&nbsp; 🤖 **Бот:** [@yandex_ml_handbook_rag_bot](https://t.me/yandex_ml_handbook_rag_bot)

---

## Содержание

- [Структура репозитория](#структура-репозитория)
- [Датасет](#датасет)
- [Чанки](#чанки)
- [Эмбеддинги и индекс](#эмбеддинги-и-индекс)
- [Retrieval](#retrieval)
- [Метрики](#метрики)

---

## Структура репозитория

  ```
  .
  ├── core/          # Ядро системы (LLM, retriever, embedder, FAISS)
  ├── evaluation/    # Метрики и валидация
  ├── models/        # Pydantic-модели данных
  ├── prompts/       # Промпты для LLM
  └── utils/         # Вспомогательные утилиты
  ```

---

## Датасет

Исходные данные — набор `.md` файлов, спарсенных с сайта учебника. Код парсинга: [`parse_handbook.py`](parse_handbook.py).

<details>
<summary>Структура <code>handbook/</code></summary>

  ```
  handbook/
  ├── 1_vvedenie/
  │   ├── 1.1_ob-etoi-knige.md
  │   ├── 1.2_pervye-shagi.md
  │   └── 1.3_mashinnoe-obuchenie.md
  ├── 2_klassicheskoe-obuchenie-s-uchitelem/
  │   ├── 2.1_vvedenie.md
  │   ├── 2.2_lineinye-modeli.md
  │   ├── 2.3_metricheskie-metody.md
  │   ├── 2.4_reshaiushchie-derev-ia.md
  │   ├── 2.5_ansambli-v-mashinnom-obuchenii.md
  │   ├── 2.6_gradientnyi-busting.md
  │   └── 2.7_zakliuchenie.md
  ├── 3_otsenka-kachestva-modelei/
  ├── 4_veroiatnostnye-modeli/
  ├── 5_glubinnoe-obuchenie-vvedenie/
  ├── 6_glubinnoe-obuchenie-arkhitektury/
  ├── 7_glubinnoe-obuchenie-praktika/
  ├── 8_generativnye-modeli/
  ├── 9_rekomendatel-nye-sistemy/
  ├── 10_prakticheskie-glavy/
  ├── 11_vzaimodeistvie-so-sredoi/
  ├── 12_teoriia-ml/
  ├── 13_teoriia-glubokogo-obucheniia/
  ├── 14_optimizatsiia-v-ml/
  ├── 15_onlain-obuchenie-i-stokhasticheskaia-optimizatsiia/
  ├── 16_teormin/
  └── index.json
  ```

</details>

---

## Чанки

Датасет нарезан на чанки размером **800 токенов** с перекрытием **15%**.  
Токенайзер: `MistralTokenizer.v3()`. Код: [`build_chunks_me.py`](build_chunks_me.py)

Пример чанка:

  ```json
  {
    "chunk_id": "1.1_ob-etoi-knige:0",
    "doc_id": "1.1_ob-etoi-knige",
    "chunk_index": 0,
    "source_url": "https://education.yandex.ru/handbook/ml/article/about",
    "path": "handbook/1_vvedenie/1.1_ob-etoi-knige.md",
    "chapter_num": "1",
    "chapter_title": "Введение",
    "article_num": "1.1",
    "article_title": "Об этой книге",
    "text": "...",
    "token_start": 0,
    "token_end": 800,
    "token_count": 800
  }
  ```

<details>
<summary>Полный список файлов <code>chunks/</code></summary>

  ```
  chunks/
  ├── 1.1_ob-etoi-knige.jsonl
  ├── 1.2_pervye-shagi.jsonl
  ├── 1.3_mashinnoe-obuchenie.jsonl
  ├── 2.1_vvedenie.jsonl
  ├── 2.2_lineinye-modeli.jsonl
  ├── 2.3_metricheskie-metody.jsonl
  ├── 2.4_reshaiushchie-derev-ia.jsonl
  ├── 2.5_ansambli-v-mashinnom-obuchenii.jsonl
  ├── 2.6_gradientnyi-busting.jsonl
  ├── 2.7_zakliuchenie.jsonl
  ├── 3.1_vvedenie.jsonl
  ├── 3.2_metriki-klassifikatsii-i-regressii.jsonl
  ├── 3.3_kross-validatsiia.jsonl
  ├── 3.4_podbor-giperparametrov.jsonl
  ├── 3.5_zakliuchenie.jsonl
  ├── 4.1_veroiatnostnyi-podkhod-v-ml.jsonl
  ├── 4.2_eksponentsial-nyi-klass-raspredelenii-i-printsip-maksimal-noi-entropii.jsonl
  ├── 4.3_obobshchionnye-lineinye-modeli.jsonl
  ├── 4.4_kak-otsenivat-veroiatnosti.jsonl
  ├── 4.5_generativnyi-podkhod-k-klassifikatsii.jsonl
  ├── 4.6_baiesovskii-podkhod-k-otsenivaniiu.jsonl
  ├── 4.7_modeli-s-latentnymi-peremennymi.jsonl
  ├── 5.1_neironnye-seti.jsonl
  ├── 5.2_pervoe-znakomstvo-s-polnosviaznymi-neirosetiami.jsonl
  ├── 5.3_metod-obratnogo-rasprostraneniia-oshibki.jsonl
  ├── 5.4_tonkosti-obucheniia.jsonl
  ├── 6.1_sviortochnye-neiroseti.jsonl
  ├── 6.2_neiroseti-dlia-raboty-s-posledovatel-nostiami.jsonl
  ├── 6.3_transformery.jsonl
  ├── 6.4_grafovye-neironnye-seti.jsonl
  ├── 6.5_neiroseti-dlia-oblakov-tochek.jsonl
  ├── 7.1_obuchenie-predstavlenii.jsonl
  ├── 7.2_distilliatsiia-znanii.jsonl
  ├── 8.1_vvedenie-v-generativnoe-modelirovanie.jsonl
  ├── 8.2_variational-autoencoder-vae.jsonl
  ├── 8.3_generativno-sostiazatel-nye-seti-gan.jsonl
  ├── 8.4_normalizuiushchie-potoki.jsonl
  ├── 8.5_diffuzionnye-modeli.jsonl
  ├── 8.6_iazykovye-modeli.jsonl
  ├── 9.1_vvedenie-v-rekomendatel-nye-sistemy.jsonl
  ├── 9.2_rekomendatsii-na-osnove-matrichnykh-razlozhenii.jsonl
  ├── 9.3_kontentnye-rekomendatsii.jsonl
  ├── 9.4_khoroshie-svoistva-rekomendatel-nykh-sistem.jsonl
  ├── 10.1_klasterizatsiia.jsonl
  ├── 10.2_vremennye-riady.jsonl
  ├── 10.3_analitika-vremennykh-riadov.jsonl
  ├── 10.4_modeli-vida-arima.jsonl
  ├── 10.5_zadacha-ranzhirovaniia.jsonl
  ├── 11.1_obuchenie-s-podkrepleniem.jsonl
  ├── 11.2_kraudsorsing.jsonl
  ├── 12.1_bias-variance-decomposition.jsonl
  ├── 13.1_vvedenie-v-teoriiu-glubokogo-obucheniia.jsonl
  ├── 13.2_obobshchaiushchaia-sposobnost-klassicheskaia-teoriia.jsonl
  ├── 13.3_pac-baiesovskie-otsenki-riska.jsonl
  ├── 13.4_seti-beskonechnoi-shiriny.jsonl
  ├── 13.5_landshaft-funktsii-poter.jsonl
  ├── 13.6_implicit-bias.jsonl
  ├── 14.1_optimizatsiia-v-ml.jsonl
  ├── 14.2_proksimal-nye-metody.jsonl
  ├── 14.3_metody-vtorogo-poriadka.jsonl
  ├── 14.4_skhodimost-sgd.jsonl
  ├── 15.1_vvedenie-v-onlain-obuchenie.jsonl
  ├── 15.2_adaptivnyi-ftrl.jsonl
  ├── 15.3_reguliarizatsiia-v-onlain-obuchenii.jsonl
  ├── 15.4_metody-optimizatsii-v-deep-learning.jsonl
  ├── 16.1_matrichnoe-differentsirovanie.jsonl
  ├── 16.2_matrichnaia-faktorizatsiia.jsonl
  ├── 16.3_veroiatnostnye-raspredeleniia.jsonl
  ├── 16.4_mnogomernye-raspredeleniia.jsonl
  ├── 16.5_nezavisimost-i-uslovnye-raspredeleniia-veroiatnostei.jsonl
  ├── 16.6_parametricheskie-otsenki.jsonl
  └── 16.7_entropiia-i-semeistvo-eksponentsial-nykh-raspredelenii.jsonl
  ```

</details>

---

## Эмбеддинги и индекс

- **Эмбеддер:** `BAAI/bge-m3` (косинусная близость)
- **Хранилище:** FAISS
- Код: [`build_index_me.py`](build_index_me.py)

---

## Retrieval

Реализованы три подхода (см. [`core/retriever.py`](core/retriever.py)):

| Ретривер | Описание |
|---|---|
| `SimpleRetriever` | Эмбеддинг вопроса → поиск по FAISS |
| `BM25Retriever` | Гибрид BM25 + косинусная близость |
| `HyDERetriever` | Генерация гипотетического ответа (HyDE) → эмбеддинг → поиск по FAISS |

---

## Метрики

Оценка проводилась на синтетическом датасете ([`eval_dataset.jsonl`](eval_dataset.jsonl)) из вопросов для ML-собеседований. Датасет собран с помощью Claude Sonnet 4.5 и Claude Opus 4.5.

**Retrieval**

| Ретривер | Recall@10 | MRR@10 |
|---|:---:|:---:|
| SimpleRetriever | 0.520 | 0.642 |
| **HyDERetriever** | **0.571** | **0.711** |
| BM25Retriever | 0.435 | 0.601 |

**Answer** (LLM-as-a-Judge, судья: `anthropic/claude-sonnet-4.6`)

| Ретривер | Correctness | Groundedness |
|---|:---:|:---:|
| SimpleRetriever | 0.758 | 0.891 |
| **HyDERetriever** | **0.766** | **0.906** |
| BM25Retriever | 0.747 | 0.883 |

**Refusal**

| Ретривер | Overall Acc | Answer Acc | Refusal Acc |
|---|:---:|:---:|:---:|
| SimpleRetriever | **0.955** | **1.000** | **0.769** |
| HyDERetriever | 0.939 | **1.000** | 0.692 |
| BM25Retriever | **0.955** | **1.000** | **0.769** |
