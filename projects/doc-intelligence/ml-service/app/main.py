"""ML service entry point.   cd ml-service && uvicorn app.main:app --port 8200   (docs at /docs)

This is the Python tier of the MERN + Python design. The browser never calls it: the Node
server (server/) does, over HTTP on the private network. It holds the models (embedder,
re-ranker, reader, summariser, generator, our fine-tuned NER + classifier) and the search index.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from .config import get_settings
from .engine import get_engine
from .routers import index, qa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(get_engine)        # load embedder + rebuild indexes before serving
    yield


app = FastAPI(title="Document Intelligence ML service", version="2.0.0", lifespan=lifespan,
              description="Index documents, search them, answer questions with cited sources.")
app.include_router(index.router)
app.include_router(qa.router)


@app.get("/health", tags=["system"])
def health():
    eng, s = get_engine(), get_settings()
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() and s.device != "cpu" else "cpu",
            "embedder": s.embedder, "chunks_indexed": eng.dense.index.ntotal,
            "documents_indexed": len(set(eng.chunk_doc.values())), "models_loaded": eng.loaded_models(),
            "generator_enabled": s.enable_generator,
            "finetuned_models": {m: (s.models_dir / m).exists() for m in ("ner", "classifier")}}


@app.get("/metrics", tags=["system"])
def metrics():
    f = get_settings().metrics_file
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
