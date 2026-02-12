import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.retriever import (
    BM25Retriever,
    HyDERetriever,
    HyDERetrieverWithQuestion,
    SimpleRetriever,
)
from evaluation.answer import run_correctness_test, run_groundedness_test
from evaluation.refusal import run_refusal_test
from evaluation.retrieval import run_mrr_test, run_recall_test

retrievers = [
    HyDERetriever(), 
    ]

for retriever in retrievers:
    print(f"Retriever: {retriever.__class__.__name__}")

    print("Retrieval metrics:")
    print(f"Recall: {run_recall_test(10, retriever):.2f}")
    print(f"MRR: {run_mrr_test(10, retriever):.2f}")
    print("-" * 20)
    
    print("Answer metrics:")
    print(f"Correctness: {run_correctness_test(10, retriever):.2f}")
    print(f"Groundedness: {run_groundedness_test(10, retriever):.2f}")
    print("-" * 20)
    
    print("Refusal metrics:")
    overall_accuracy, answer_accuracy, refusal_accuracy = run_refusal_test(10, retriever)
    print(f"Overall accuracy: {overall_accuracy:.2f}")
    print(f"Answer accuracy: {answer_accuracy:.2f}")
    print(f"Refusal accuracy: {refusal_accuracy:.2f}")
    print("-" * 20)
    print('=' * 20)