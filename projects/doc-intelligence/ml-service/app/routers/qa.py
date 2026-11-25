"""Search and question answering. The chat history is stored by the Node server (MongoDB) and
sent with each question, so this service stays stateless apart from the index."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from ..engine import get_engine, hit_json
from ..schemas import AskIn, AskOut, SearchIn, SearchOut

router = APIRouter(tags=["search & QA"])


@router.post("/search", response_model=SearchOut)
async def search(body: SearchIn):
    """Semantic / keyword / hybrid search, optionally re-ranked. Model work runs in a thread pool
    so the async event loop stays free for other requests."""
    eng = get_engine()
    hits, timing = await run_in_threadpool(eng.search, body.query, body.mode, body.rerank, body.top_k,
                                           body.document_ids)
    return SearchOut(hits=[hit_json(h, i) for i, h in enumerate(hits, 1)], timing=timing)


@router.post("/ask", response_model=AskOut)
async def ask(body: AskIn):
    history = [t.model_dump() for t in body.history]
    try:
        result = await run_in_threadpool(get_engine().ask, body.question, body.mode, body.document_ids, history)
    except RuntimeError as e:                               # e.g. generator disabled on this machine
        raise HTTPException(503, str(e))
    return AskOut(**result)
