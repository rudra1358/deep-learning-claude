"""FastAPI application entry point.

Run locally:   uvicorn app.main:app --port 8100        (from the backend/ folder)
API docs:      http://localhost:8100/docs
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

import onnxruntime as ort
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import init_db
from .models_registry import get_registry
from .routers import detect, runs, stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    init_db(s.database_url)
    reg = get_registry()
    if s.default_model in reg.available():
        reg.get(s.default_model)                 # load + warm up before the first request arrives
    yield


app = FastAPI(title="Vision Analytics API", version="1.0.0", lifespan=lifespan,
              description="Object detection, multi-object tracking and visual analytics.")
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(detect.router)
app.include_router(runs.router)
app.include_router(stream.router)


@app.get("/api/health", tags=["system"])
def health():
    return {"status": "ok", "providers": ort.get_available_providers(),
            "models": get_registry().available(), "default_model": get_settings().default_model}


@app.get("/api/models", tags=["system"])
def models():
    return get_registry().describe()


@app.get("/api/metrics", tags=["system"])
def metrics():
    """Offline evaluation results written by training/evaluate.py (mAP, per-class AP, latency)."""
    mf = get_settings().metrics_file
    return json.loads(mf.read_text()) if mf.exists() else {"models": {}}
