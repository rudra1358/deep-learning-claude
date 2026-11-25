"""Upload, list, inspect and delete documents."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ingest.extract import SUPPORTED

from .. import worker
from ..config import get_settings
from ..db import ChunkRow, Document, get_db
from ..engine import get_engine
from ..schemas import DocumentOut, PageOut

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _doc_or_404(db: Session, doc_id: int) -> Document:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "document not found")
    return doc


@router.post("", response_model=list[DocumentOut], status_code=202)
async def upload(files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    """Save the files, create 'queued' rows and hand them to the background worker (202 Accepted)."""
    s = get_settings()
    created = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in SUPPORTED:
            raise HTTPException(415, f"{f.filename}: unsupported type; use {sorted(SUPPORTED)}")
        path = s.data_dir / "uploads" / f"{uuid.uuid4().hex}{ext}"
        size = 0
        with path.open("wb") as out:                       # stream to disk, never the whole file in RAM
            while chunk := await f.read(1 << 20):
                size += len(chunk)
                if size > s.max_upload_mb * 2**20:
                    out.close()
                    path.unlink(missing_ok=True)
                    raise HTTPException(413, f"{f.filename}: larger than {s.max_upload_mb} MB")
                out.write(chunk)
        doc = Document(filename=f.filename or path.name, path=str(path))
        db.add(doc)
        db.commit()
        created.append(doc)
    for doc in created:
        worker.submit(doc.id)
    return created


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    return db.scalars(select(Document).order_by(Document.created_at.desc(), Document.id.desc())).all()


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, db: Session = Depends(get_db)):
    return _doc_or_404(db, doc_id)


@router.get("/{doc_id}/pages", response_model=list[PageOut])
def get_pages(doc_id: int, db: Session = Depends(get_db)):
    return [PageOut(number=p.number, text=p.text) for p in _doc_or_404(db, doc_id).pages]


@router.get("/{doc_id}/file")
def get_file(doc_id: int, db: Session = Depends(get_db)):
    doc = _doc_or_404(db, doc_id)
    if not Path(doc.path).exists():
        raise HTTPException(404, "original file missing")
    return FileResponse(doc.path, filename=doc.filename)


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    doc = _doc_or_404(db, doc_id)
    if doc.status in ("queued", "processing") or doc.enrich_status == "running":
        raise HTTPException(409, "document is still being processed")
    chunk_ids = list(db.scalars(select(ChunkRow.id).where(ChunkRow.document_id == doc_id)))
    get_engine().remove_chunks(chunk_ids)                   # remove from FAISS + BM25 first
    Path(doc.path).unlink(missing_ok=True)
    db.delete(doc)
    db.commit()
