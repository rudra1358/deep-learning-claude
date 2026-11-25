"""Request / response shapes (the API contract, also shown at /docs)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utc(v: datetime) -> datetime:
    return v if v.tzinfo else v.replace(tzinfo=timezone.utc)   # SQLite drops the timezone


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    filename: str
    status: str
    enrich_status: str
    error: str | None
    num_pages: int
    num_chunks: int
    label: str | None
    label_score: float | None
    entities: list | None
    summary: str | None
    created_at: datetime
    _tz = field_validator("created_at")(_utc)


class PageOut(BaseModel):
    number: int
    text: str


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    rerank: bool = True
    top_k: int = Field(5, ge=1, le=50)
    document_ids: list[int] | None = None


class Source(BaseModel):
    n: int
    chunk_id: int
    document_id: int
    filename: str
    page_start: int
    page_end: int
    char_start: int
    score: float
    ranks: dict
    text: str


class SearchOut(BaseModel):
    hits: list[Source]
    timing: dict


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    mode: Literal["extractive", "generative"] = "extractive"
    document_ids: list[int] | None = None
    session_id: int | None = None        # continue a conversation; None = start a new one


class AskOut(BaseModel):
    session_id: int
    answer: str | None
    found: bool
    mode: str
    cited: list[int] = []
    confidence: float | None = None
    highlight: dict | None = None        # {source: n, start, end} - exact answer span inside source n
    suggestion: dict | None = None       # extractive only: low-confidence candidate {text, confidence, threshold, highlight}
    gated: bool = False                  # generative only: the reader found no answer, so the LLM was skipped
    sources: list[Source]
    timing: dict


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    role: str
    content: str
    sources: list | None
    meta: dict | None
    created_at: datetime
    _tz = field_validator("created_at")(_utc)


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    created_at: datetime
    _tz = field_validator("created_at")(_utc)


class SessionDetail(SessionOut):
    messages: list[MessageOut]
