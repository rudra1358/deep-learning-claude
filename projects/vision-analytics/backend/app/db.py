"""Database layer: SQLAlchemy 2.0 ORM models and session handling.

Works with SQLite (default, zero setup) and PostgreSQL (Docker Compose) - only the
connection URL changes. Tables:

  runs          one row per analysed image / video / live stream session
  detections    individual boxes for image runs
  tracks        one row per tracked object in a video/stream run (first/last frame, label)
  events        line-crossing events (track, label, direction, time)
  count_samples per-second occupancy per class -> "counts over time" charts
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from .config import get_settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(10), index=True)          # image | video | stream
    model: Mapped[str] = mapped_column(String(80))
    source_name: Mapped[str] = mapped_column(String(255), default="")
    media_path: Mapped[str | None] = mapped_column(String(500), nullable=True)   # stored upload (videos)
    status: Mapped[str] = mapped_column(String(12), default="done", index=True)  # queued|running|done|failed
    progress: Mapped[float] = mapped_column(Float, default=1.0)
    error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    frames: Mapped[int] = mapped_column(Integer, default=0)
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    conf: Mapped[float] = mapped_column(Float, default=0.25)
    line: Mapped[list | None] = mapped_column(JSON, nullable=True)      # [x1, y1, x2, y2] normalised 0..1
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    detections: Mapped[list[Detection]] = relationship(back_populates="run", cascade="all, delete-orphan")
    tracks: Mapped[list[TrackRow]] = relationship(back_populates="run", cascade="all, delete-orphan")
    events: Mapped[list[EventRow]] = relationship(back_populates="run", cascade="all, delete-orphan")
    samples: Mapped[list[CountSample]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(64), index=True)
    score: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    run: Mapped[Run] = relationship(back_populates="detections")


class TrackRow(Base):
    __tablename__ = "tracks"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String(64), index=True)
    first_frame: Mapped[int] = mapped_column(Integer)
    last_frame: Mapped[int] = mapped_column(Integer)
    max_score: Mapped[float] = mapped_column(Float)
    run: Mapped[Run] = relationship(back_populates="tracks")


class EventRow(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String(64))
    direction: Mapped[str] = mapped_column(String(4))
    t_seconds: Mapped[float] = mapped_column(Float)
    run: Mapped[Run] = relationship(back_populates="events")


class CountSample(Base):
    __tablename__ = "count_samples"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    t_seconds: Mapped[float] = mapped_column(Float)
    label: Mapped[str] = mapped_column(String(64))
    count: Mapped[int] = mapped_column(Integer)
    run: Mapped[Run] = relationship(back_populates="samples")


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def init_db(url: str | None = None) -> None:
    global _engine, _SessionLocal
    url = url or get_settings().database_url
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    _engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def _fk_on(dbapi_conn, _):  # SQLite needs this for ON DELETE CASCADE
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")   # readers don't block the video worker's writes
            cur.close()
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(_engine, expire_on_commit=False)


def session_factory() -> sessionmaker[Session]:
    if _SessionLocal is None:
        init_db()
    assert _SessionLocal is not None
    return _SessionLocal


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    db = session_factory()()
    try:
        yield db
    finally:
        db.close()
