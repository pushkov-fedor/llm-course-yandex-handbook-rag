import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import prompts.judge as judge
import prompts.rag as rag
from core.llm import generate_response
from core.retriever import Retriever, SimpleRetriever
from models.dataset_item import DatasetItem
from models.judge_response import AnswerJudgeResponse
from utils.dataset import read_eval_dataset
from utils.messages import get_rag_user_prompt


def calculate_correctness(dataset_item: DatasetItem, k: int, retriever: Retriever) -> int:
    question: str = dataset_item.question
    
    top_k_chunks = retriever.retrieve(question, k)

    user_prompt = get_rag_user_prompt(question, top_k_chunks)

    rag_response = generate_response(system_prompt=rag.SYSTEM_PROMPT, user_prompt=user_prompt)

    judge_prompt = judge.CORRECTNESS_USER_TEMPLATE.format(
        question=question,
        reference_answer=dataset_item.answer,
        model_answer=rag_response
    )

    judge_response_raw = generate_response(
        system_prompt=judge.CORRECTNESS_SYSTEM_PROMPT, 
        user_prompt=judge_prompt, 
        model=config.JUDGE_MODEL, 
        response_format=AnswerJudgeResponse.get_response_format())

    judge_response = AnswerJudgeResponse(**json.loads(judge_response_raw))
    return judge_response.score

def calculate_groundedness(dataset_item: DatasetItem, k: int, retriever: Retriever) -> int:
    question: str = dataset_item.question
    top_k_chunks = retriever.retrieve(question, k)

    context = "\n\n".join([chunk.to_string() for chunk in top_k_chunks])
    user_prompt = get_rag_user_prompt(question, top_k_chunks)

    rag_response = generate_response(system_prompt=rag.SYSTEM_PROMPT, user_prompt=user_prompt)

    judge_prompt = judge.GROUNDEDNESS_USER_TEMPLATE.format(
        context=context,
        question=question,
        answer=rag_response
    )

    judge_response_raw = generate_response(
        system_prompt=judge.GROUNDEDNESS_SYSTEM_PROMPT, 
        user_prompt=judge_prompt, 
        model=config.JUDGE_MODEL, 
        response_format=AnswerJudgeResponse.get_response_format())

    judge_response = AnswerJudgeResponse(**json.loads(judge_response_raw))
    return judge_response.score

def run(k: int, func: Callable[[DatasetItem, int, Retriever], int], retriever: Retriever, max_workers: int = 10):
    dataset_items = read_eval_dataset()

    scores = [0] * len(dataset_items)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, dataset_item in enumerate(dataset_items):
            future = executor.submit(func, dataset_item, k, retriever)
            future_to_idx[future] = idx

        for future in tqdm(as_completed(future_to_idx), total=len(dataset_items), desc=f"Evaluating {func.__name__}"):
            idx = future_to_idx[future]
            scores[idx] = future.result()

    return sum(scores) / (5 * len(scores))

def run_correctness_test(k: int, retriever: Retriever) -> float:
    return run(k, calculate_correctness, retriever)

def run_groundedness_test(k: int, retriever: Retriever) -> float:
    return run(k, calculate_groundedness, retriever)

if __name__ == "__main__":
    print('Calculating correctness...')
    correctness = run_correctness_test(10, SimpleRetriever())
    print(f"Correctness: {correctness}")

    print('Calculating groundedness...')
    groundedness = run_groundedness_test(10, SimpleRetriever())
    print(f"Groundedness: {groundedness}")