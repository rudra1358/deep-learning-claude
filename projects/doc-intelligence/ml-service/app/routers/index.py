"""Indexing endpoints, called by the Node server's background jobs (server/src/jobs.js).

  POST   /index/documents       extract -> chunk -> embed -> add to FAISS + BM25; returns the pages
  DELETE /index/documents/{id}  remove a document's chunks from both indexes
  POST   /enrich                topic label, entities, summary for an indexed document

These are plain `def` (not `async def`): FastAPI runs them in a worker thread, so a long
indexing job does not block /search or /ask requests arriving at the same time.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from ingest.extract import SUPPORTED

from ..engine import get_engine
from ..schemas import EnrichIn, EnrichOut, IndexIn, IndexOut, PageOut

router = APIRouter(tags=["indexing"])


@router.post("/index/documents", response_model=IndexOut)
def index_document(body: IndexIn):
    path = Path(body.path)
    if path.suffix.lower() not in SUPPORTED:
        raise HTTPException(415, f"unsupported type; use {sorted(SUPPORTED)}")
    if not path.exists():
        raise HTTPException(404, "file not found")
    try:
        pages, n = get_engine().index_document(body.document_id, body.filename, str(path))
    except ValueError as e:                                  # e.g. a scanned PDF with no text layer
        raise HTTPException(422, str(e))
    return IndexOut(document_id=body.document_id, pages=[PageOut(number=p.number, text=p.text) for p in pages],
                    num_chunks=n)


@router.delete("/index/documents/{document_id}")
def remove_document(document_id: str):
    return {"removed_chunks": get_engine().remove_document(document_id)}


@router.post("/enrich", response_model=EnrichOut)
def enrich(body: EnrichIn):
    try:
        return get_engine().enrich(body.document_id, body.text)
    except KeyError:
        raise HTTPException(404, "document is not indexed")
