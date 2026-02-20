import json
import os
import sys

from tqdm import tqdm

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
    SimpleRetriever(),
    HyDERetriever(), 
    BM25Retriever(),
    ]

metrics_list = []

for retriever in tqdm(retrievers, desc="Validating retrievers"):
    metrics = {}

    recall = run_recall_test(10, retriever)
    mrr = run_mrr_test(10, retriever)
    
    correctness = run_correctness_test(10, retriever)
    groundedness = run_groundedness_test(10, retriever)

    overall_accuracy, answer_accuracy, refusal_accuracy = run_refusal_test(10, retriever)

    metrics["Retriever"] = retriever.__class__.__name__
    metrics["Recall"] = recall
    metrics["MRR"] = mrr
    metrics["Correctness"] = correctness
    metrics["Groundedness"] = groundedness
    metrics["Overall Accuracy"] = overall_accuracy
    metrics["Answer Accuracy"] = answer_accuracy
    metrics["Refusal Accuracy"] = refusal_accuracy

    metrics_list.append(metrics)

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_results.json")
with open(output_path, "w") as f:
    json.dump(metrics_list, f, indent=4)