# Vision Analytics — real-time object detection, tracking & counting

Detect, track and count objects in images, uploaded videos and a live webcam, with a dashboard for
history, analytics and model comparison.

* **Detector:** YOLO11 (nano and small), COCO-pretrained and fine-tuned on **VisDrone** (drone/CCTV traffic),
  served as **ONNX** with ONNX Runtime (CUDA or CPU).
* **Tracker:** ByteTrack-style multi-object tracker written from scratch (Kalman filter + Hungarian matching
  + two-stage association with low-confidence detections).
* **Analytics:** line-crossing counts (in/out), occupancy over time, dwell time, unique objects.
* **Backend:** FastAPI (HTTP + WebSocket), SQLAlchemy (SQLite or PostgreSQL), background video jobs.
* **Frontend:** Next.js dashboard — live webcam overlay, image & video analysis, run playback with
  tracking overlay, history search & CSV export, model comparison, metrics.
* **Ops:** Docker Compose (CPU and GPU profiles), pytest suite with a generated fake model, GitHub Actions CI.

Every box shown in the app comes from model inference. All numbers in the docs are measured — see
[docs/RESULTS.md](docs/RESULTS.md).

**New here / preparing for an interview?** Read in this order:
1. [docs/HOW_IT_WAS_BUILT.md](docs/HOW_IT_WAS_BUILT.md) — the ideas and design decisions, in build order
2. [docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md) — every file explained, following one frame through the code
3. [docs/RESULTS.md](docs/RESULTS.md) — the measured numbers you can quote
4. [docs/INTERVIEW_QA.md](docs/INTERVIEW_QA.md) — practice questions with model answers

## Repository layout

```
vision-analytics/
├── backend/
│   ├── inference/        # pure NumPy/OpenCV + ONNX Runtime — no PyTorch needed to serve
│   │   ├── preprocess.py   letterbox resize
│   │   ├── postprocess.py  YOLO output decoding, IoU, (class-wise) NMS, rescaling
│   │   ├── detector.py     ONNX Runtime session, provider selection, timing
│   │   ├── kalman.py       constant-velocity Kalman filter for boxes
│   │   ├── tracker.py      ByteTrack-style tracker
│   │   └── analytics.py    line crossings, occupancy, dwell time
│   ├── app/              # FastAPI: main.py, routers/ (detect, runs, stream), db.py, pipeline.py, video_jobs.py
│   ├── tests/            # unit + API tests (fake ONNX model generated in conftest.py)
│   ├── Dockerfile  Dockerfile.gpu  requirements.txt
├── training/
│   ├── configs/          # experiments.yaml (all hyperparameters), dataset yamls
│   ├── prepare_data.py   # download VisDrone + COCO subset, convert, EDA
│   ├── evaluate.py       # COCO baseline, domain gap, fine-tuned evaluation
│   ├── train.py          # config-driven fine-tuning
│   ├── export_onnx.py    # ONNX export + parity check + latency (fp32/fp16/int8)
│   ├── run_all.ps1       # the whole pipeline in order
│   └── reports/          # metrics.json, EDA, logs (read by the API's /api/metrics)
├── frontend/             # Next.js dashboard (see frontend/README.md)
├── scripts/              # dev_api.py, fetch_demo_assets.py, benchmark_pipeline.py, make_results.py
├── models/               # exported .onnx models served by the API (generated)
├── docs/                 # HOW_IT_WAS_BUILT.md, INTERVIEW_QA.md, RESULTS.md
└── docker-compose.yml
```

## Quick start (local)

Requirements: Python 3.12, Node 20+, optionally an NVIDIA GPU.

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or CPU wheels
.venv/Scripts/pip install -r training/requirements.txt -r backend/requirements-dev.txt
```

Get models — either the full pipeline (downloads ~3.5 GB of data, trains for a few hours on a laptop GPU):

```bash
powershell -ExecutionPolicy Bypass -File training/run_all.ps1
```

…or just the pretrained COCO models (minutes): `python training/export_onnx.py --export-only` downloads the
COCO weights and writes `models/yolo11n-coco.onnx` and `models/yolo11s-coco.onnx`.

**Step-by-step local setup for both projects (one shared environment): see [../README.md](../README.md).**

Run the app:

```bash
.venv/Scripts/python scripts/dev_api.py            # API on http://localhost:8100  (docs at /docs)
cd frontend && npm install && npm run build && npm start   # dashboard on http://localhost:3100
```

Tests: `cd backend && ../.venv/Scripts/python -m pytest` (no GPU or trained weights needed).

## Docker

```bash
docker compose up --build                                   # PostgreSQL + API (CPU) + dashboard
docker compose --profile gpu up --build db api-gpu web      # NVIDIA GPU (needs NVIDIA Container Toolkit)
```

The images copy `models/*.onnx` and `training/reports/` — create them first (see above).
Then check the running stack end to end (API container + PostgreSQL + dashboard):

```bash
python scripts/docker_smoke.py
```

CI (`.github/workflows/ci.yml`, job `docker`) does the same on every push: builds the images, starts the
stack with a generated test model, and runs `scripts/docker_smoke.py`.

**Windows:** Docker Desktop needs WSL 2. If `wsl -l -v` says WSL is not installed, run `wsl --install` in an
administrator terminal and restart, then start Docker Desktop.

## Configuration

Environment variables (prefix `VA_`): `VA_DATABASE_URL`, `VA_MODELS_DIR`, `VA_DATA_DIR`, `VA_DEVICE`
(`auto|cuda|cpu`), `VA_DEFAULT_MODEL`, `VA_MAX_UPLOAD_MB`, `VA_MAX_VIDEO_SECONDS`, `VA_CORS_ORIGINS`.
Frontend: `NEXT_PUBLIC_API_URL` (default `http://localhost:8100`).

## Which model to use

| model | trained on | use for |
|---|---|---|
| `yolo11n-coco`, `yolo11s-coco` | COCO (80 everyday classes) | webcam, street-level photos, general scenes (default) |
| `yolo11n-visdrone`, `yolo11s-visdrone` | VisDrone (10 traffic classes) | drone / elevated CCTV traffic footage |

## Notes

* Windows: install `onnxruntime-gpu[cuda,cudnn]` — recent ONNX Runtime GPU builds target CUDA 13, while
  PyTorch wheels ship CUDA 12 DLLs; the extras install matching runtime wheels. The API logs a warning if
  CUDA was requested but ONNX Runtime fell back to CPU.
* Windows + 16 GB RAM: keep `workers` ≤ 2 in `experiments.yaml` (each DataLoader worker is a process that
  loads the CUDA libraries; 4 workers exhausted the page file during development).
* Licences: Ultralytics YOLO is AGPL-3.0; VisDrone and COCO have their own dataset terms (research use).
