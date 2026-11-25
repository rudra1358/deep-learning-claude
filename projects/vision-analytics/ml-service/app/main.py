"""ML service entry point (Python / FastAPI).

This service does ONLY the deep-learning work: run the ONNX detector, track objects, compute
analytics. It has no database. The Node/Express server (../server) owns MongoDB, the upload
history and everything the browser talks to; it calls this service over HTTP/WebSocket:

    React (client/)  ->  Express + MongoDB (server/)  ->  this service (ml-service/)

Run locally:   python scripts/dev_api.py            (or: uvicorn app.main:app --port 8100 from ml-service/)
API docs:      http://localhost:8100/docs
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

import onnxruntime as ort
from fastapi import FastAPI

from .config import get_settings
from .models_registry import get_registry
from .routers import detect, jobs, stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    reg = get_registry()
    if s.default_model in reg.available():
        reg.get(s.default_model)                 # load + warm up before the first request arrives
    yield


# No CORS middleware: browsers never call this service directly - only the Express server does.
app = FastAPI(title="Vision Analytics ML service", version="2.0.0", lifespan=lifespan,
              description="Object detection, multi-object tracking and visual analytics (no storage).")
app.include_router(detect.router)
app.include_router(jobs.router)
app.include_router(stream.router)


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "providers": ort.get_available_providers(),
            "models": get_registry().available(), "default_model": get_settings().default_model}


@app.get("/models", tags=["system"])
def models():
    return get_registry().describe()


@app.get("/metrics", tags=["system"])
def metrics():
    """Offline evaluation results written by training/evaluate.py (mAP, per-class AP, latency)."""
    mf = get_settings().metrics_file
    return json.loads(mf.read_text()) if mf.exists() else {"models": {}}
