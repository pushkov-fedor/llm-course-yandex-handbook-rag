from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    source_url: str
    path: str
    chapter_num: str
    chapter_title: str
    article_num: str
    article_title: str
    text: str
    token_start: int
    token_end: int
    token_count: int

    def to_string(self) -> str:
        return f"""
            Глава: {self.chapter_title}
            Статья: {self.article_title}
            Текст: {self.text}
        """