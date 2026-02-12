import os
import sys
from collections.abc import Callable

from core.retriever import Retriever, SimpleRetriever

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from models.dataset_item import DatasetItem
from utils.dataset import read_eval_dataset


def calculate_recall(dataset_item: DatasetItem, k: int, retriever: Retriever) -> float:
    question = dataset_item.question
    relevant_chunk_ids = dataset_item.relevant_chunk_ids

    if relevant_chunk_ids is None or len(relevant_chunk_ids) == 0:
        return 0.0

    top_k_chunks = retriever.retrieve(question, k)
    top_k_chunk_ids = [chunk.chunk_id for chunk in top_k_chunks]

    correct_chunks_ids = set(relevant_chunk_ids).intersection(set(top_k_chunk_ids))

    return len(correct_chunks_ids) / len(relevant_chunk_ids)

def calculate_mrr(dataset_item: DatasetItem, k: int, retriever: Retriever) -> float:
    question = dataset_item.question
    relevant_chunk_ids = dataset_item.relevant_chunk_ids

    if relevant_chunk_ids is None or len(relevant_chunk_ids) == 0:
        return 0.0

    relevant_chunk_ids = set(relevant_chunk_ids)

    top_k_chunks = retriever.retrieve(question, k)
    top_k_chunk_ids = [chunk.chunk_id for chunk in top_k_chunks]

    for i, chunk in enumerate(top_k_chunk_ids):
        if chunk in relevant_chunk_ids:
            return 1 / (i + 1)
    
    return 0.0

def run(k: int, func: Callable[[DatasetItem, int, Retriever], float], retriever: Retriever, max_workers: int = 10):
    dataset_items = read_eval_dataset()

    scores = [0.0] * len(dataset_items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, dataset_item in enumerate(dataset_items):
            future = executor.submit(func, dataset_item, k, retriever)
            future_to_idx[future] = idx

        for future in tqdm(as_completed(future_to_idx), total=len(dataset_items), desc=f"Evaluating {func.__name__}"):
            idx = future_to_idx[future]
            scores[idx] = future.result()

    return sum(scores) / len(scores)

def run_recall_test(k: int, retriever: Retriever) -> float:
    return run(k, calculate_recall, retriever)

def run_mrr_test(k: int, retriever: Retriever) -> float:
    return run(k, calculate_mrr, retriever)

if __name__ == "__main__":
    print('Calculating recall...')
    recall = run(10, calculate_recall, SimpleRetriever())
    print(f"Recall: {recall}")

    print('Calculating mrr...')
    mrr = run(10, calculate_mrr, SimpleRetriever())
    print(f"MRR: {mrr}")