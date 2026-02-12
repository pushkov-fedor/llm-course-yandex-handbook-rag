import json
import os
from pathlib import Path

from models.chunk import Chunk


def load_chunks() -> list[Chunk]:
    """
    Загружает чанки из файла или директории.
    
    Returns:
        list[Chunk]: Список всех чанков
    """
    path = os.path.join(os.path.dirname(__file__), "../chunks")

    chunks = []
    
    # Если path указывает на директорию, загружаем все .jsonl файлы из неё
    if os.path.isdir(path):
        jsonl_files = sorted(Path(path).glob('*.jsonl'))
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(Chunk(**json.loads(line)))
    # Иначе загружаем как обычный файл (для обратной совместимости)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(Chunk(**json.loads(line)))
    
    return chunks

chunks = load_chunks()

def get_chunks_by_indices(indices: list[int]) -> list[Chunk]:
    return [chunks[i] for i in indices]