"""Progress-sync API for Deep Learning University.

The web app works fully offline with localStorage; when NEXT_PUBLIC_API_URL points
here, the client also pushes/pulls its progress so it survives across devices.

Storage: SQLite by default (zero setup). Set DATABASE_URL to a PostgreSQL URL
(e.g. postgresql+psycopg://user:pass@host/db) for production.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("dlu")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./progress.db")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
MAX_PAYLOAD_BYTES = 256 * 1024  # progress documents are small; reject anything abusive
LEARNER_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})


class Base(DeclarativeBase):
    pass


class ProgressRow(Base):
    __tablename__ = "progress"
    learner_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    data: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


Base.metadata.create_all(engine)


class QuizScore(BaseModel):
    score: int = Field(ge=0)
    total: int = Field(gt=0)
    at: float


class Progress(BaseModel):
    """Mirror of the client's ProgressState (web/lib/progress.ts)."""

    learnerId: str
    completed: dict[str, float] = {}
    quizzes: dict[str, QuizScore] = {}
    bookmarks: list[str] = []
    projects: dict[str, list[str]] = {}
    activeDays: list[str] = []
    lastVisited: str | None = None


def merge(a: Progress, b: Progress) -> Progress:
    """Union-merge two progress documents; never lose a completion or a better score."""
    quizzes = dict(a.quizzes)
    for k, v in b.quizzes.items():
        if k not in quizzes or v.score / v.total > quizzes[k].score / quizzes[k].total:
            quizzes[k] = v
    projects = {k: sorted(set(a.projects.get(k, [])) | set(v)) for k, v in b.projects.items()}
    for k, v in a.projects.items():
        projects.setdefault(k, v)
    return Progress(
        learnerId=a.learnerId,
        completed={**b.completed, **a.completed},
        quizzes=quizzes,
        bookmarks=sorted(set(a.bookmarks) | set(b.bookmarks)),
        projects=projects,
        activeDays=sorted(set(a.activeDays) | set(b.activeDays)),
        lastVisited=b.lastVisited or a.lastVisited,
    )


app = FastAPI(title="Deep Learning University API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "PUT"], allow_headers=["Content-Type"])


def _check_id(learner_id: str) -> None:
    if not LEARNER_ID.match(learner_id):
        raise HTTPException(status_code=422, detail="invalid learner id")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/progress/{learner_id}", response_model=Progress)
def get_progress(learner_id: str) -> Progress:
    _check_id(learner_id)
    with Session(engine) as s:
        row = s.get(ProgressRow, learner_id)
        if row is None:
            raise HTTPException(status_code=404, detail="no progress stored for this learner")
        return Progress.model_validate_json(row.data)


@app.put("/api/progress/{learner_id}", response_model=Progress)
def put_progress(learner_id: str, body: Progress) -> Progress:
    _check_id(learner_id)
    if body.learnerId != learner_id:
        raise HTTPException(status_code=422, detail="learnerId in body does not match the URL")
    payload = body.model_dump_json()
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise HTTPException(status_code=413, detail="progress document too large")
    with Session(engine) as s:
        row = s.get(ProgressRow, learner_id)
        merged = merge(body, Progress.model_validate_json(row.data)) if row else body
        data = merged.model_dump_json()
        now = datetime.now(timezone.utc)
        if row:
            row.data, row.updated_at = data, now
        else:
            s.add(ProgressRow(learner_id=learner_id, data=data, updated_at=now))
        s.commit()
        log.info("saved progress for %s (%d completed)", learner_id[:8], len(merged.completed))
        return merged


@app.get("/api/stats")
def stats() -> dict[str, int]:
    """Aggregate, anonymous counts (no learner data exposed)."""
    with Session(engine) as s:
        rows = s.scalars(select(ProgressRow)).all()
        completions = sum(len(json.loads(r.data).get("completed", {})) for r in rows)
        return {"learners": len(rows), "completions": completions}
