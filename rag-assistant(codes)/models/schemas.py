from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    sessionId: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("sessionId", "message")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class SourceChunk(BaseModel):
    title: str
    chunk_id: str
    source_document: str
    similarity: float
    text: str


class ChatResponse(BaseModel):
    sessionId: str
    answer: str
    sources: list[SourceChunk]
    fallback: bool = False
    createdAt: datetime


class ErrorResponse(BaseModel):
    error: dict[str, str]


class HealthResponse(BaseModel):
    status: Literal["healthy"]
