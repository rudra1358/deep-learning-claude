"""Start the Python ML service for local development from any folder:
    python scripts/dev_api.py [--device cpu]

The Node server (server/, port 4200) talks to it on http://127.0.0.1:8200."""
import argparse
import os
import sys
from pathlib import Path

import uvicorn

BACKEND = Path(__file__).resolve().parents[1] / "ml-service"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=os.environ.get("DI_DEVICE", "auto"))
    ap.add_argument("--port", type=int, default=8200)
    a = ap.parse_args()
    os.environ["DI_DEVICE"] = a.device
    os.chdir(BACKEND)
    sys.path.insert(0, str(BACKEND))
    uvicorn.run("app.main:app", host="127.0.0.1", port=a.port)
