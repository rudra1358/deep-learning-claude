"""Request / response shapes of the ML service (the contract with the Node server, also shown at /docs).

Document ids are strings: they are MongoDB ObjectIds created by the server. This service never
creates documents - it only indexes the file it is given and answers questions over the index.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IndexIn(BaseModel):
    document_id: str = Field(min_length=1)
    filename: str
    path: str                            # where the server saved the upload (shared data folder)


class PageOut(BaseModel):
    number: int
    text: str


class IndexOut(BaseModel):
    document_id: str
    pages: list[PageOut]
    num_chunks: int


class EnrichIn(BaseModel):
    document_id: str = Field(min_length=1)
    text: str = ""                       # full document text (the summariser reads it); "" = join the chunks


class EnrichOut(BaseModel):
    label: str | None
    label_score: float | None
    entities: list | None
    summary: str | None


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    rerank: bool = True
    top_k: int = Field(5, ge=1, le=50)
    document_ids: list[str] | None = None


class Source(BaseModel):
    n: int
    chunk_id: int
    document_id: str
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


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    mode: Literal["extractive", "generative"] = "extractive"
    document_ids: list[str] | None = None
    history: list[Turn] = []             # earlier turns of the chat (stored in MongoDB by the server)


class AskOut(BaseModel):
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
