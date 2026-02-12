import prompts.rag as rag
from models.chunk import Chunk


def get_rag_user_prompt(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join([chunk.to_string() for chunk in chunks])
    return rag.USER_TEMPLATE.format(question=question, context=context)