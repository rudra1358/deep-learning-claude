"""Background ingestion. Upload returns immediately; this worker does the slow part.

Stage 1 - make it searchable (status: queued -> processing -> ready):
    extract pages -> chunk -> save pages + chunks to the DB -> embed chunks -> add to FAISS + BM25
Stage 2 - enrich it (enrich_status: pending -> running -> done):
    topic label (our classifier), entities (our NER model), summary (DistilBART map-reduce)

The two stages run on TWO separate queues, each with one worker thread. Originally one worker did
both; then a slow summary (its model was still downloading) left every newer upload stuck in
"queued" - not even searchable - for minutes. Now a document becomes searchable as soon as the
index worker reaches it, whatever the enrichment worker is doing. One worker per queue keeps GPU
use predictable. At scale: a real queue (Celery/RQ + Redis) with N workers per stage.
"""
from __future__ import annotations

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

from ingest.extract import extract

from .db import ChunkRow, Document, PageRow, session_factory
from .engine import get_engine

log = logging.getLogger("worker")
_index_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="index")
_enrich_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="enrich")


def submit(document_id: int) -> None:
    """New upload: index it; when that succeeds the index worker hands it to the enrichment queue."""
    _index_pool.submit(_run_index, document_id)


def recover() -> int:
    """Called at startup. Jobs live in memory, so if the API stopped mid-job a document would stay
    'queued'/'processing' (or enrichment 'running') forever. Reset those and run them again."""
    from sqlalchemy import and_, or_, select
    with session_factory()() as db:
        # "pending" counts too: a document that was WAITING in the in-memory enrich queue when the API
        # stopped is lost just like one that was running (found in live testing - 3 docs stayed pending)
        stuck = db.scalars(select(Document).where(or_(
            Document.status.in_(("queued", "processing")),
            and_(Document.status == "ready", Document.enrich_status.in_(("pending", "running")))))).all()
        to_index, to_enrich = [], []
        for doc in stuck:
            if doc.status != "ready":                    # half-indexed: drop partial pages/chunks, start over
                doc.pages, doc.chunks, doc.status = [], [], "queued"
                to_index.append(doc.id)
            else:                                        # searchable already, only enrichment was cut off
                to_enrich.append(doc.id)
            doc.enrich_status = "pending"
        db.commit()
    for i in to_index:
        submit(i)
    for i in to_enrich:
        _enrich_pool.submit(_run_enrich, i)
    return len(to_index) + len(to_enrich)


def _run_index(document_id: int) -> None:
    with session_factory()() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            return
        try:
            _index(db, doc)
        except Exception as exc:  # noqa: BLE001 - record any failure on the document
            log.error("indexing %s failed: %s", document_id, traceback.format_exc())
            db.rollback()
            doc = db.get(Document, document_id)
            doc.status, doc.enrich_status, doc.error = "failed", "skipped", str(exc)[:500]
            db.commit()
            return
    _enrich_pool.submit(_run_enrich, document_id)


def _run_enrich(document_id: int) -> None:
    with session_factory()() as db:
        doc = db.get(Document, document_id)
        if doc is None or doc.status != "ready":         # deleted in the meantime, or failed
            return
        try:
            _enrich(db, doc)
        except Exception as exc:  # noqa: BLE001 - enrichment is optional; search still works
            log.error("enrichment %s failed: %s", document_id, traceback.format_exc())
            db.rollback()
            doc = db.get(Document, document_id)
            doc.enrich_status, doc.error = "failed", f"enrichment: {exc}"[:500]
            db.commit()


def _index(db, doc: Document) -> None:
    engine = get_engine()
    doc.status = "processing"
    db.commit()
    pages = extract(doc.path)
    chunks = engine.make_chunks(pages)
    doc.pages = [PageRow(number=p.number, text=p.text) for p in pages]
    doc.chunks = [ChunkRow(index=c.index, text=c.text, page_start=c.page_start, page_end=c.page_end,
                           char_start=c.char_start) for c in chunks]
    doc.num_pages, doc.num_chunks = len(pages), len(chunks)
    db.commit()                                            # assigns chunk ids (= vector ids)
    engine.add_chunks(doc.chunks)
    doc.status = "ready"
    db.commit()
    log.info("indexed %s: %d pages, %d chunks", doc.filename, len(pages), len(chunks))


def _enrich(db, doc: Document) -> None:
    engine = get_engine()
    doc.enrich_status = "running"
    db.commit()
    texts = [c.text for c in doc.chunks]
    if (engine.s.models_dir / "classifier").exists():
        doc.label, doc.label_score, _ = engine.model("classifier").classify_document(texts)
        doc.label_score = round(doc.label_score, 4)
        db.commit()
    if (engine.s.models_dir / "ner").exists():
        doc.entities = engine.model("ner").document_entities(texts)
        db.commit()
    full_text = "\n".join(p.text for p in doc.pages)
    doc.summary = engine.model("summarizer").summarize(full_text)
    doc.enrich_status = "done"
    db.commit()
