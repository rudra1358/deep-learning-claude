"""Search, question answering and chat history."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ..db import ChatMessage, ChatSession, get_db
from ..engine import get_engine, hit_json
from ..schemas import AskIn, AskOut, SearchIn, SearchOut, SessionDetail, SessionOut

router = APIRouter(prefix="/api", tags=["search & QA"])


@router.post("/search", response_model=SearchOut)
async def search(body: SearchIn):
    """Semantic / keyword / hybrid search, optionally re-ranked. Model work runs in a thread pool
    so the async event loop stays free for other requests."""
    eng = get_engine()
    hits, timing = await run_in_threadpool(eng.search, body.query, body.mode, body.rerank, body.top_k,
                                           body.document_ids)
    return SearchOut(hits=[hit_json(h, i) for i, h in enumerate(hits, 1)], timing=timing)


@router.post("/ask", response_model=AskOut)
async def ask(body: AskIn, db: Session = Depends(get_db)):
    if body.session_id is not None:
        session = db.get(ChatSession, body.session_id)
        if session is None:
            raise HTTPException(404, "chat session not found")
    else:
        session = ChatSession(title=body.question[:80])
        db.add(session)
        db.commit()
    history = [{"role": m.role, "content": m.content} for m in session.messages]
    try:
        result = await run_in_threadpool(get_engine().ask, body.question, body.mode, body.document_ids, history)
    except RuntimeError as e:                               # e.g. generator disabled on this machine
        raise HTTPException(503, str(e))
    content = result["answer"] or "No answer found in the documents."
    db.add_all([
        ChatMessage(session_id=session.id, role="user", content=body.question),
        ChatMessage(session_id=session.id, role="assistant", content=content, sources=result["sources"],
                    meta={k: result.get(k) for k in ("mode", "found", "cited", "highlight", "timing", "confidence", "suggestion", "gated")}),
    ])
    db.commit()
    return AskOut(session_id=session.id, **result)


@router.get("/chats", response_model=list[SessionOut])
def list_chats(db: Session = Depends(get_db)):
    return db.scalars(select(ChatSession).order_by(ChatSession.id.desc())).all()


@router.get("/chats/{session_id}", response_model=SessionDetail)
def get_chat(session_id: int, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(404, "chat session not found")
    return session


@router.delete("/chats/{session_id}", status_code=204)
def delete_chat(session_id: int, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(404, "chat session not found")
    db.delete(session)
    db.commit()
