import json
import os
from pathlib import Path


def load_chunks(path: str) -> list[dict]:
    """
    Загружает чанки из файла или директории.
    
    Args:
        path: Путь к файлу chunks.jsonl или к директории chunks/
        
    Returns:
        list[dict]: Список всех чанков
    """
    chunks = []
    
    # Если path указывает на директорию, загружаем все .jsonl файлы из неё
    if os.path.isdir(path):
        jsonl_files = sorted(Path(path).glob('*.jsonl'))
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
    # Иначе загружаем как обычный файл (для обратной совместимости)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    
    return chunks