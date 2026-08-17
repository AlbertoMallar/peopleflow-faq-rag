from pydantic import BaseModel, ConfigDict


class ChunkRelated(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    section_title: str
    relevance_score: float
    content: str


class RAGResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_question: str
    system_answer: str
    chunks_related: list[ChunkRelated]