# Vision Analytics — real-time object detection, tracking & counting

Detect, track and count objects in images, uploaded videos and a live webcam, with a dashboard for
history, analytics and model comparison.

* **Detector:** YOLO11 (nano and small), COCO-pretrained and fine-tuned on **VisDrone** (drone/CCTV traffic),
  served as **ONNX** with ONNX Runtime (CUDA or CPU).
* **Tracker:** ByteTrack-style multi-object tracker written from scratch (Kalman filter + Hungarian matching
  + two-stage association with low-confidence detections).
* **Analytics:** line-crossing counts (in/out), occupancy over time, dwell time, unique objects.
* **Web app (MERN):** **R**eact dashboard (Vite, plain JavaScript) → **N**ode + **E**xpress API → **M**ongoDB.
  Live webcam overlay, image & video analysis, run playback with tracking overlay, history search & CSV
  export, model comparison, metrics.
* **ML service (Python):** FastAPI over HTTP + WebSocket — runs the detector, tracker and analytics, and
  processes videos in a background worker. It stores nothing; the Express server owns the data.
* **Ops:** Docker Compose (CPU and GPU profiles), pytest (fake ONNX model) + Node tests (fake ML service,
  real MongoDB), GitHub Actions CI.

```
browser (React, client/)  ──/api + WebSocket──▶  Express server (server/)  ──HTTP + WebSocket──▶  ML service (ml-service/, Python)
                                                      │
                                                   MongoDB  (runs, detections, tracks, events, counts)
```

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
├── ml-service/           # Python: the deep learning (no database)
│   ├── inference/        # pure NumPy/OpenCV + ONNX Runtime — no PyTorch needed to serve
│   │   ├── preprocess.py   letterbox resize
│   │   ├── postprocess.py  YOLO output decoding, IoU, (class-wise) NMS, rescaling
│   │   ├── detector.py     ONNX Runtime session, provider selection, timing
│   │   ├── kalman.py       constant-velocity Kalman filter for boxes
│   │   ├── tracker.py      ByteTrack-style tracker
│   │   └── analytics.py    line crossings, occupancy, dwell time
│   ├── app/              # FastAPI: main.py, routers/ (detect, jobs, stream), pipeline.py, video_jobs.py
│   ├── tests/            # unit + API tests (fake ONNX model generated in conftest.py)
│   ├── Dockerfile  Dockerfile.gpu  requirements.txt
├── server/               # Node + Express + MongoDB: the API the browser uses (see server/README.md)
│   ├── src/              # index.js, app.js, models/Run.js, routes/, jobs.js (video jobs), stream.js (WebSocket relay)
│   ├── tests/            # API tests: real MongoDB + a fake ML service
│   └── Dockerfile        # builds the client too and serves it
├── client/               # React + Vite dashboard, plain JavaScript (see client/README.md)
├── training/
│   ├── configs/          # experiments.yaml (all hyperparameters), dataset yamls
│   ├── prepare_data.py   # download VisDrone + COCO subset, convert, EDA
│   ├── evaluate.py       # COCO baseline, domain gap, fine-tuned evaluation
│   ├── train.py          # config-driven fine-tuning
│   ├── export_onnx.py    # ONNX export + parity check + latency (fp32/fp16/int8)
│   ├── run_all.ps1       # the whole pipeline in order
│   └── reports/          # metrics.json, EDA, logs (read by the API's /api/metrics)
├── scripts/              # dev_api.py, fetch_demo_assets.py, benchmark_pipeline.py, make_results.py
├── models/               # exported .onnx models served by the API (generated)
├── docs/                 # HOW_IT_WAS_BUILT.md, INTERVIEW_QA.md, RESULTS.md
└── docker-compose.yml
```

## Quick start (local)

Requirements: Python 3.12, Node 20+, MongoDB (local install or `docker run -p 27017:27017 mongo:7`),
optionally an NVIDIA GPU.

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or CPU wheels
.venv/Scripts/pip install -r training/requirements.txt -r ml-service/requirements-dev.txt
```

Get models — either the full pipeline (downloads ~3.5 GB of data, trains for a few hours on a laptop GPU):

```bash
powershell -ExecutionPolicy Bypass -File training/run_all.ps1
```

…or just the pretrained COCO models (minutes): `python training/export_onnx.py --export-only` downloads the
COCO weights and writes `models/yolo11n-coco.onnx` and `models/yolo11s-coco.onnx`.

**Step-by-step local setup for both projects (one shared environment): see [../README.md](../README.md).**

Run the app — three terminals:

```bash
.venv/Scripts/python scripts/dev_api.py            # 1. Python ML service on :8100 (docs at http://localhost:8100/docs)
cd server && npm install && npm start              # 2. Express API on :4100 (needs MongoDB running)
cd client && npm install && npm run dev            # 3. React dashboard on http://localhost:3100
```

Open **http://localhost:3100**. (Production-style instead of 3: `cd client && npm run build`, then the
Express server serves the dashboard itself at http://localhost:4100.)

Tests:

```bash
cd ml-service && ../.venv/Scripts/python -m pytest      # Python (no GPU or trained weights needed)
cd server && npm test                                   # Node (needs MongoDB; uses a fake ML service)
```

## Docker

```bash
docker compose up --build                                   # MongoDB + ML service (CPU) + server/dashboard
docker compose --profile gpu up --build mongo ml-gpu server # NVIDIA GPU (needs NVIDIA Container Toolkit)
```

The images copy `models/*.onnx` and `training/reports/` — create them first (see above).
Open http://localhost:4100. Then check the running stack end to end (server + MongoDB + ML service):

```bash
python scripts/docker_smoke.py
```

CI (`.github/workflows/ci.yml`, job `docker`) does the same on every push: builds the images, starts the
stack with a generated test model, and runs `scripts/docker_smoke.py`.

**Windows:** Docker Desktop needs WSL 2. If `wsl -l -v` says WSL is not installed, run `wsl --install` in an
administrator terminal and restart, then start Docker Desktop.

## Configuration

ML service (prefix `VA_`): `VA_MODELS_DIR`, `VA_DATA_DIR`, `VA_DEVICE` (`auto|cuda|cpu`), `VA_DEFAULT_MODEL`,
`VA_MAX_UPLOAD_MB`, `VA_MAX_VIDEO_SECONDS`.
Server (or `server/.env`): `PORT` (4100), `MONGO_URL` (`mongodb://127.0.0.1:27017/vision`), `ML_URL`
(`http://127.0.0.1:8100`), `DATA_DIR` (`../data` — must be the same folder as `VA_DATA_DIR`), `MAX_UPLOAD_MB`.
Client: `VITE_API_URL` only if the API is on another host (by default it uses relative `/api` URLs).

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
