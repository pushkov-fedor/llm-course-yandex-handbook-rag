#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /home/fedor/Study/llm-course/build_chunks_me.py

import json
import os
import re
from math import floor
from typing import Any

import tiktoken
from tqdm import tqdm

from parse_handbook import TocItem

enc = tiktoken.get_encoding("cl100k_base")
output_file = 'chunks.jsonl'

def get_page_content(path: str) -> str:
    stream = open(path, 'r', encoding='utf-8')
    return stream.read()

doc_path = 'handbook/1_vvedenie/1.1_ob-etoi-knige.md'

def process_file(item: TocItem, chunk_size: int = 800, overlap_ratio: float = 0.15) -> list[str]:
    file_path = item.file_path
    chapter_num = item.chapter_num
    chapter_title = item.chapter_title
    article_num = item.article_num
    article_title = item.article_title
    url = item.url
    
    doc_id = file_path.split('/')[-1].replace('.md', '')
    text = get_page_content(file_path)
    
    tokens = enc.encode(text)
    total = len(tokens)
    if total == 0:
        return []

    overlap = int(floor(chunk_size * overlap_ratio))
    overlap = max(0, min(overlap, chunk_size - 1))
    step = chunk_size - overlap

    chunks = []
    chunk_index = 0
    for start in range(0, total, step):
        end = min(start + chunk_size, total)
        text = enc.decode(tokens[start:end])
        token_count = len(tokens[start:end])
        token_start = start
        token_end = end
        chunk_id = f"{doc_id}:{chunk_index}"

        chunks.append({
            "chunk_id": chunk_id, 
            "doc_id": doc_id, 
            "chunk_index": chunk_index, 
            "source_url": url, 
            "path": file_path,
            "chapter_num": chapter_num,
            "chapter_title": chapter_title,
            "article_num": article_num,
            "article_title": article_title,
            "text": text, 
            "token_start": token_start, 
            "token_end": token_end, 
            "token_count": token_count
            })
        chunk_index += 1
        if end == total:
            break

    return chunks



def generate_chunks() -> None:
    index_path = 'handbook/index.json'
    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)

    with open(output_file, 'w', encoding='utf-8'):
        pass

    for item_dict in tqdm(index['items'], desc="Processing items"):
        item = TocItem(**item_dict)
        chunks = process_file(item, 800, 0.15)
        
        with open(output_file, 'a', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

generate_chunks()