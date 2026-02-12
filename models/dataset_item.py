from pydantic import BaseModel


class DatasetItem(BaseModel):
    id: int
    question: str
    relevant_docs: list[str]
    relevant_chunk_ids: list[str] | None = None
    type: str
    should_answer: bool
    answer: str | None = None