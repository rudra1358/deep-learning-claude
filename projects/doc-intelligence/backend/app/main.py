"""FastAPI entry point.   cd backend && uvicorn app.main:app --port 8200   (docs at /docs)"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from . import worker
from .config import get_settings
from .db import init_db
from .engine import get_engine
from .routers import documents, qa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(get_settings().database_url)
    await run_in_threadpool(get_engine)        # load embedder + rebuild indexes before serving
    n = worker.recover()                       # restart jobs interrupted by a previous shutdown
    if n:
        logging.getLogger("startup").info("re-queued %d interrupted document(s)", n)
    yield


app = FastAPI(title="Document Intelligence API", version="1.0.0", lifespan=lifespan,
              description="Upload documents, search them, ask questions with cited answers.")
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins, allow_methods=["*"], allow_headers=["*"])
app.include_router(documents.router)
app.include_router(qa.router)


@app.get("/api/health", tags=["system"])
def health():
    eng, s = get_engine(), get_settings()
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() and s.device != "cpu" else "cpu",
            "embedder": s.embedder, "chunks_indexed": eng.dense.index.ntotal, "models_loaded": eng.loaded_models(),
            "generator_enabled": s.enable_generator,
            "finetuned_models": {m: (s.models_dir / m).exists() for m in ("ner", "classifier")}}


@app.get("/api/metrics", tags=["system"])
def metrics():
    f = get_settings().metrics_file
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
