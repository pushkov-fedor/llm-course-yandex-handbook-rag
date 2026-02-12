from abc import ABC, abstractmethod

import numpy as np
from rank_bm25 import BM25Okapi

import prompts.hyDE as hyDE
from core.chunks import chunks, get_chunks_by_indices
from core.embedder import encode
from core.faiss_index import search_in_index
from core.llm import generate_response
from models.chunk import Chunk


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, question: str, k: int) -> list[Chunk]:
        pass

class SimpleRetriever(Retriever):
    def retrieve(self, question: str, k: int) -> list[Chunk]:
        embedding = encode(question)
        _, indices = search_in_index(embedding, k)
        top_k_chunks = get_chunks_by_indices(indices[0])
        return top_k_chunks

class HyDERetriever(SimpleRetriever):
    def retrieve(self, question: str, k: int) -> list[Chunk]:
        hypothehical_answer_response = generate_response(
            system_prompt=hyDE.SYSTEM_PROMPT,
            user_prompt=question,
        )
        return super().retrieve(hypothehical_answer_response, k)

class HyDERetrieverWithQuestion(SimpleRetriever):
    def retrieve(self, question: str, k: int) -> list[Chunk]:
        hypothehical_answer_response = generate_response(
            system_prompt=hyDE.SYSTEM_PROMPT,
            user_prompt=question,
        )
        question_prompt = f"Question: {question}\nHypothetical answer: {hypothehical_answer_response}"
        return super().retrieve(question_prompt, k)

class BM25Retriever(Retriever):
    def __init__(self, alpha: float = 0.3):
        self._alpha = alpha
        tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, question: str, k: int) -> list[Chunk]:
        tokenized_query = question.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        bm25_max = bm25_scores.max()
        if bm25_max > 0:
            bm25_scores = bm25_scores / bm25_max

        embedding = encode(question)
        faiss_scores, faiss_indices = search_in_index(embedding, len(chunks))
        emb_scores = np.zeros(len(chunks))
        emb_scores[faiss_indices[0]] = faiss_scores[0]
        emb_max = emb_scores.max()
        if emb_max > 0:
            emb_scores = emb_scores / emb_max

        combined = self._alpha * bm25_scores + (1 - self._alpha) * emb_scores
        top_k_indices = np.argsort(combined)[::-1][:k]
        return [chunks[i] for i in top_k_indices]