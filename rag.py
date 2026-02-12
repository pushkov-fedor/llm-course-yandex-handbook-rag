import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prompts.rag as rag
from core.llm import generate_response
from core.retriever import HyDERetriever, Retriever
from utils.messages import get_rag_user_prompt


def run(question: str, k: int, retriever: Retriever = HyDERetriever()) -> str:
    top_k_chunks = retriever.retrieve(question, k)
    user_prompt = get_rag_user_prompt(question, top_k_chunks)
    rag_response = generate_response(system_prompt=rag.SYSTEM_PROMPT, user_prompt=user_prompt)
    return rag_response
