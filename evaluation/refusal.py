import json
import os
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from core.retriever import Retriever, SimpleRetriever

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import prompts.judge as judge
import prompts.rag as rag
from core.llm import generate_response
from models.dataset_item import DatasetItem
from models.judge_response import RefusalJudgeResponse
from utils.dataset import read_eval_dataset
from utils.messages import get_rag_user_prompt


def calculate_refusal(dataset_item: DatasetItem, k: int, retriever: Retriever) -> bool:
    question: str = dataset_item.question
    top_k_chunks = retriever.retrieve(question, k)
    user_prompt = get_rag_user_prompt(question, top_k_chunks)
    rag_response = generate_response(
        system_prompt=rag.SYSTEM_PROMPT, 
        user_prompt=user_prompt
    )

    judge_prompt = judge.REFUSAL_JUDGE_USER_TEMPLATE.format(
        question=question,
        answer=rag_response
    )

    judge_response_raw = generate_response(
        system_prompt=judge.REFUSAL_JUDGE_SYSTEM_PROMPT, 
        user_prompt=judge_prompt, 
        model=config.JUDGE_MODEL, 
        response_format=RefusalJudgeResponse.get_response_format())

    judge_response = RefusalJudgeResponse(**json.loads(judge_response_raw))
    return judge_response.refused

def run(k: int, func: Callable[[DatasetItem, int, Retriever], bool], retriever: Retriever, max_workers: int = 10) -> tuple[float, float, float]:
    dataset_items = read_eval_dataset(answer_only=False)

    results: list[tuple[bool, bool]] = [(False, False)] * len(dataset_items)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, dataset_item in enumerate(dataset_items):
            future = executor.submit(func, dataset_item, k, retriever)
            future_to_idx[future] = (idx, dataset_item.should_answer)

        for future in tqdm(as_completed(future_to_idx), total=len(dataset_items), desc=f"Evaluating {func.__name__}"):
            idx, should_answer = future_to_idx[future]
            refused = future.result()
            results[idx] = (should_answer, refused)

    overall_score = sum(1 for should_answer, refused in results 
        if (should_answer and not refused) or (not should_answer and refused))
    total = len(results)
    overall_accuracy = overall_score / total

    answer_score = sum(1 for should_answer, refused in results if should_answer and not refused)
    answer_accuracy = answer_score / sum(1 for should_answer, _ in results if should_answer)

    refusal_score = sum(1 for should_answer, refused in results if not should_answer and refused)
    refusal_accuracy = refusal_score / sum(1 for should_answer, _ in results if not should_answer)

    return overall_accuracy, answer_accuracy, refusal_accuracy

def run_refusal_test(k: int, retriever: Retriever) -> tuple[float, float, float]:
    return run(k, calculate_refusal, retriever)

if __name__ == "__main__":
    print('Calculating refusal accuracy...')
    overall_accuracy, answer_accuracy, refusal_accuracy = run(10, calculate_refusal, SimpleRetriever())
    print(f"Overall accuracy: {overall_accuracy}")
    print(f"Answer accuracy: {answer_accuracy}")
    print(f"Refusal accuracy: {refusal_accuracy}")