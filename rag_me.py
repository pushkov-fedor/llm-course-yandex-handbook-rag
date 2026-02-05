# /home/fedor/Study/llm-course/rag_me.py
import faiss
from mistralai import Mistral
from sentence_transformers import SentenceTransformer

import config
from utils.load_chunks import load_chunks

llm = Mistral(api_key=config.MISTRAL_API_KEY)
embedder = SentenceTransformer(config.EMBEDDER_MODEL, device="cpu")
index = faiss.read_index(config.INDEX_FILE)

chunks = load_chunks("chunks")

def answer_question(question: str) -> str:
    embedding = embedder.encode([question], normalize_embeddings=True)
    similarities, indices = index.search(embedding, k=5)

    print('Similarities: ', similarities[0])

    top_k_chunks = [chunks[i] for i in indices[0]]

    context = [f"--- Источник {i}: {chunk['article_num']}. {chunk['article_title']} ---\n{chunk['text']}" for i, chunk in enumerate(top_k_chunks, 1)]
    user_request = config.RAG_USER_TEMPLATE.format(question=question, context='\n\n'.join(context))

    print(user_request + '\n\n' + '-' * 100 + '\n\n')

    response = llm.chat.complete(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": config.RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("=" * 60)
    print("RAG-помощник по учебнику ML от Яндекса")
    print("=" * 60)
    
    while True:
        question = input("\n🔍 Введите вопрос (или 'q' для выхода): ").strip()
        if question.lower() in ('q', 'quit', 'exit', ''):
            print("До свидания!")
            break
        print("\n" + "-" * 60 + "\n")
        print(answer_question(question))