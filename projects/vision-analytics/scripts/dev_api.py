"""Start the API for local development:  python scripts/dev_api.py [--device cpu|cuda|auto] [--reload]

Equivalent to `cd backend && uvicorn app.main:app --port 8100`, but works from any folder.
"""
import argparse
import os
import sys
from pathlib import Path

import uvicorn

BACKEND = Path(__file__).resolve().parents[1] / "backend"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=os.environ.get("VA_DEVICE", "auto"))
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()
    os.environ["VA_DEVICE"] = a.device
    os.chdir(BACKEND)
    sys.path.insert(0, str(BACKEND))
    uvicorn.run("app.main:app", host="127.0.0.1", port=a.port, reload=a.reload,
                reload_dirs=[str(BACKEND)] if a.reload else None)
