#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /home/fedor/Study/llm-course/build_chunks.py
"""
Нарезка markdown статей на чанки по токенам Mistral.

Использование:
    python build_chunks.py --input_dir handbook --output chunks.jsonl
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


def load_index_metadata(input_dir: Path) -> Dict[str, dict]:
    """
    Загрузить метаданные из index.json.
    
    Args:
        input_dir: Директория с handbook
        
    Returns:
        Словарь source_url -> {chapter_num, chapter_title, article_num, article_title}
    """
    index_path = input_dir / "index.json"
    if not index_path.exists():
        print(f"Предупреждение: {index_path} не найден, метаданные не будут добавлены")
        return {}
    
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
    except Exception as e:
        print(f"Ошибка чтения {index_path}: {e}")
        return {}
    
    metadata = {}
    for item in index_data.get("items", []):
        url = item.get("url")
        if url:
            metadata[url] = {
                "chapter_num": item.get("chapter_num", ""),
                "chapter_title": item.get("chapter_title", ""),
                "article_num": item.get("article_num", ""),
                "article_title": item.get("article_title", ""),
            }
    
    print(f"Загружено метаданных для {len(metadata)} статей из index.json")
    return metadata


def get_mistral_tokenizer():
    """
    Получить токенайзер Mistral через transformers.
    Использует AutoTokenizer с моделью mistralai/Mistral-7B-v0.1.
    """
    try:
        from transformers import AutoTokenizer
        
        # Используем базовую модель Mistral (токенайзер одинаковый для всех версий)
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            trust_remote_code=True
        )
        return tokenizer
    except ImportError:
        raise ImportError(
            "Не удалось импортировать transformers.\n"
            "Установите пакет:\n"
            "  pip install transformers\n"
        )
    except Exception as e:
        raise RuntimeError(
            f"Ошибка загрузки токенайзера Mistral: {e}\n"
            "Убедитесь, что есть доступ к интернету для загрузки модели.\n"
            "Или установите transformers и sentencepiece:\n"
            "  pip install transformers sentencepiece\n"
        )


class MistralTokenizerAdapter:
    """
    Адаптер для унификации интерфейса токенайзера.
    Работает с transformers AutoTokenizer.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> List[int]:
        """Закодировать текст в список токенов."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: List[int]) -> str:
        """Декодировать список токенов в текст."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def iter_md_files(input_dir: Path) -> Iterator[Path]:
    """
    Рекурсивно итерировать по всем .md файлам в директории.
    Пропускает index.json.
    """
    for md_file in input_dir.rglob("*.md"):
        if md_file.name == "index.json":
            continue
        yield md_file


def extract_source_url(text: str) -> Optional[str]:
    """
    Извлечь source_url из строки вида '_Source: https://..._'.
    
    Args:
        text: Текст файла
        
    Returns:
        URL или None, если не найден
    """
    # Ищем строку вида "_Source: https://..._" или "_Source: http://..._"
    pattern = r"_Source:\s*(https?://[^\s_]+)_"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def strip_header(text: str) -> str:
    """
    Убрать шапку из начала файла:
    - заголовок # ...
    - строку _Source: ..._
    - разделитель ---
    
    Args:
        text: Исходный текст
        
    Returns:
        Текст без шапки
    """
    lines = text.split("\n")
    i = 0

    # Пропускаем заголовок # ...
    if i < len(lines) and lines[i].strip().startswith("# "):
        i += 1
        # Пропускаем пустую строку после заголовка
        if i < len(lines) and not lines[i].strip():
            i += 1

    # Пропускаем строку _Source: ..._
    if i < len(lines) and "_Source:" in lines[i]:
        i += 1
        # Пропускаем пустую строку после source
        if i < len(lines) and not lines[i].strip():
            i += 1

    # Пропускаем разделитель ---
    if i < len(lines) and lines[i].strip() == "---":
        i += 1
        # Пропускаем пустую строку после разделителя
        if i < len(lines) and not lines[i].strip():
            i += 1

    return "\n".join(lines[i:]).lstrip()


def chunk_by_tokens(
    text: str,
    tokenizer: MistralTokenizerAdapter,
    chunk_size: int = 800,
    overlap_ratio: float = 0.15,
) -> List[Tuple[str, int, int, int]]:
    """
    Нарезать текст на чанки по токенам с перекрытием.
    
    Args:
        text: Исходный текст
        tokenizer: Адаптер токенайзера
        chunk_size: Размер чанка в токенах
        overlap_ratio: Коэффициент перекрытия (0.15 = 15%)
        
    Returns:
        Список кортежей: (chunk_text, token_start, token_end, token_count)
    """
    # Кодируем весь текст в токены
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    
    if total_tokens == 0:
        return []
    
    overlap_tokens = round(chunk_size * overlap_ratio)
    chunks = []
    pos = 0
    
    while pos < total_tokens:
        # Определяем конец текущего чанка
        end_pos = min(pos + chunk_size, total_tokens)
        chunk_tokens = tokens[pos:end_pos]
        
        # Декодируем чанк обратно в текст
        chunk_text = tokenizer.decode(chunk_tokens)
        
        # Проверяем минимальную длину после strip
        chunk_text_stripped = chunk_text.strip()
        if len(chunk_text_stripped) >= 50:
            token_count = len(chunk_tokens)
            chunks.append((chunk_text, pos, end_pos, token_count))
        
        # Если дошли до конца, выходим
        if end_pos >= total_tokens:
            break
        
        # Сдвигаем позицию с учетом перекрытия
        pos = end_pos - overlap_tokens
        # Защита от зацикливания
        if pos <= 0:
            pos = end_pos
    
    return chunks


def write_jsonl(output_file: Path, records: List[dict]) -> None:
    """
    Записать записи в JSON Lines формат.
    
    Args:
        output_file: Путь к выходному файлу
        records: Список словарей для записи
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Нарезка markdown статей на чанки по токенам Mistral"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./handbook",
        help="Директория с .md файлами (по умолчанию: ./handbook)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="chunks.jsonl",
        help="Выходной файл JSONL (по умолчанию: chunks.jsonl)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=800,
        help="Размер чанка в токенах (по умолчанию: 800)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.15,
        help="Коэффициент перекрытия (по умолчанию: 0.15)",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output)
    
    if not input_dir.exists():
        print(f"Ошибка: директория {input_dir} не существует")
        return 1
    
    # Инициализация токенайзера
    print("Загрузка токенайзера Mistral...")
    try:
        raw_tokenizer = get_mistral_tokenizer()
        tokenizer = MistralTokenizerAdapter(raw_tokenizer)
        print("Токенайзер успешно загружен")
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        return 1
    except Exception as e:
        print(f"Ошибка инициализации токенайзера: {e}")
        return 1
    
    # Загрузка метаданных из index.json
    metadata_by_url = load_index_metadata(input_dir)
    
    # Обработка файлов
    records = []
    doc_chunk_counts = defaultdict(int)
    missing_source_urls = []
    missing_metadata = []
    processed_docs = 0
    
    md_files = list(iter_md_files(input_dir))
    print(f"Найдено {len(md_files)} .md файлов")
    
    for md_file in sorted(md_files):
        doc_id = md_file.stem  # Имя файла без расширения
        # Относительный путь от input_dir
        input_dir_resolved = input_dir.resolve()
        md_file_resolved = md_file.resolve()
        rel_path = str(md_file_resolved.relative_to(input_dir_resolved))
        
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Ошибка чтения {md_file}: {e}")
            continue
        
        # Извлекаем source_url
        source_url = extract_source_url(content)
        if not source_url:
            missing_source_urls.append(rel_path)
            continue
        
        # Убираем шапку
        text_full = strip_header(content)
        
        if not text_full.strip():
            print(f"Предупреждение: {rel_path} пуст после удаления шапки")
            continue
        
        # Нарезаем на чанки
        chunks = chunk_by_tokens(
            text_full, tokenizer, args.chunk_size, args.overlap
        )
        
        if not chunks:
            print(f"Предупреждение: {rel_path} не дал ни одного чанка")
            continue
        
        # Получаем метаданные для этого документа
        doc_metadata = metadata_by_url.get(source_url, {})
        if not doc_metadata:
            missing_metadata.append(rel_path)
        
        # Формируем записи
        for chunk_index, (chunk_text, token_start, token_end, token_count) in enumerate(chunks):
            chunk_id = f"{doc_id}:{chunk_index}"
            
            record = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "source_url": source_url,
                "path": rel_path,
                # Метаданные из index.json
                "chapter_num": doc_metadata.get("chapter_num", ""),
                "chapter_title": doc_metadata.get("chapter_title", ""),
                "article_num": doc_metadata.get("article_num", ""),
                "article_title": doc_metadata.get("article_title", ""),
                # Текст и токены
                "text": chunk_text,
                "token_count": token_count,
                "token_start": token_start,
                "token_end": token_end,
            }
            records.append(record)
            doc_chunk_counts[doc_id] += 1
        
        processed_docs += 1
        if processed_docs % 10 == 0:
            print(f"Обработано документов: {processed_docs}")
    
    # Записываем результат
    write_jsonl(output_file, records)
    
    # Статистика
    print("\n" + "=" * 50)
    print("СТАТИСТИКА")
    print("=" * 50)
    print(f"Документов обработано: {processed_docs}")
    print(f"Всего чанков: {len(records)}")
    
    if doc_chunk_counts:
        print("\nТоп-5 документов по числу чанков:")
        top_docs = sorted(doc_chunk_counts.items(), key=lambda x: -x[1])[:5]
        for doc_id, count in top_docs:
            print(f"  {doc_id}: {count} чанков")
    
    if missing_source_urls:
        print(f"\nФайлов без source_url: {len(missing_source_urls)}")
        for path in missing_source_urls[:10]:
            print(f"  {path}")
        if len(missing_source_urls) > 10:
            print(f"  ... и ещё {len(missing_source_urls) - 10} файлов")
    
    if missing_metadata:
        print(f"\nФайлов без метаданных в index.json: {len(missing_metadata)}")
        for path in missing_metadata[:10]:
            print(f"  {path}")
        if len(missing_metadata) > 10:
            print(f"  ... и ещё {len(missing_metadata) - 10} файлов")
    
    print(f"\nРезультат сохранён в: {output_file}")
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    exit(main())
