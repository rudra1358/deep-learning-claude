# Projects — how to run them locally

Two complete, measured deep-learning systems. Each folder has its own README and a `docs/` folder
(how it was built, file-by-file code walkthrough, measured results, interview Q&A).

| project | what it does | UI | API |
|---|---|---|---|
| [vision-analytics](vision-analytics/) | Detects, tracks and counts objects in images, videos and a live webcam (YOLO11 + ByteTrack) | http://localhost:3100 | http://localhost:8100/docs |
| [doc-intelligence](doc-intelligence/) | Answers questions about your documents with the exact source highlighted, or says "not found" (RAG) | http://localhost:3200 | http://localhost:8200/docs |

Commands below are for **Windows (Git Bash or PowerShell)**. On macOS/Linux replace `.venv/Scripts/` with
`.venv/bin/`.

---

## 1. Requirements

* **Python 3.12** (3.11 also works) — `python --version`
* **Node.js 20+** — `node --version`
* **Git**
* Optional: an **NVIDIA GPU** (developed on an RTX 3050 6 GB). Everything also runs on CPU, just slower.
* Disk: ~6 GB for Python packages + ~4 GB of models downloaded on first use (more if you re-train).

## 2. One-time setup: one Python environment for both projects

From the `projects/` folder:

```bash
python -m venv .venv
```

PyTorch — pick ONE line. GPU (NVIDIA):

```bash
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

CPU only:

```bash
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then everything else for both projects (on a CPU-only machine first change the last line of
`requirements-local.txt` to `onnxruntime>=1.20`):

```bash
.venv/Scripts/pip install -r requirements-local.txt
```

Hugging Face downloads sometimes hang at 0 bytes on some networks; this setting avoids it (set it in every new
terminal, or add it to your environment variables):

```bash
export HF_HUB_DISABLE_XET=1
```

(PowerShell: `$env:HF_HUB_DISABLE_XET = "1"`.)

---

## 3. Project 1 — Vision Analytics

### 3a. Get the models (5 minutes)

The trained `.onnx` files are not stored in git. This downloads the COCO-pretrained YOLO11 weights and exports
them to ONNX:

```bash
cd vision-analytics
../.venv/Scripts/python training/export_onnx.py --export-only
```

You now have `models/yolo11n-coco.onnx` and `models/yolo11s-coco.onnx` (people, cars, 80 everyday classes —
good for the webcam and street videos). The drone-footage models (`*-visdrone`) need the full training run
(section 3d).

### 3b. Start the API (terminal 1)

```bash
cd vision-analytics
../.venv/Scripts/python scripts/dev_api.py
```

Open http://localhost:8100/docs — the interactive API documentation. `--device cpu` forces CPU.

### 3c. Start the dashboard (terminal 2)

```bash
cd vision-analytics/frontend
npm install
npm run build
npm start
```

Open **http://localhost:3100**:

* **Live** — allow the webcam, optionally draw a counting line, press Start.
* **Video** — upload a video, draw a line, watch the tracked playback and charts. A test clip:
  `../.venv/Scripts/python scripts/fetch_demo_assets.py` downloads `demo/pedestrians.avi`.
* **Image**, **Compare** (two models side by side), **History**, **Metrics**.

### 3d. Optional: reproduce training and evaluation (~6 h on a laptop GPU)

```bash
cd vision-analytics
powershell -ExecutionPolicy Bypass -File training/run_all.ps1
../.venv/Scripts/python scripts/make_results.py
```

Downloads VisDrone (~2.3 GB) + a COCO subset, runs the baselines, fine-tunes YOLO11n and YOLO11s, evaluates,
exports with parity checks and regenerates `docs/RESULTS.md`. With 16 GB RAM keep `workers: 2` in
`training/configs/experiments.yaml`.

### 3e. Tests

```bash
cd vision-analytics/backend
../../.venv/Scripts/python -m pytest
```

27 tests, no GPU or trained models needed.

---

## 4. Project 2 — Document Intelligence

### 4a. (Optional) train the two small models (~10 minutes on a GPU)

The app works without them; they add entity extraction and topic labels to each document.

```bash
cd doc-intelligence
../.venv/Scripts/python training/finetune_ner.py
../.venv/Scripts/python training/finetune_classifier.py
```

### 4b. Start the API (terminal 1)

```bash
cd doc-intelligence
../.venv/Scripts/python scripts/dev_api.py
```

The first start downloads the embedding model; the re-ranker, reader (~500 MB), summariser (~1.2 GB) and the
optional LLM (~3 GB) download the first time they are used. On a machine without a GPU, turn the LLM off
before starting: `export DI_ENABLE_GENERATOR=false`.

### 4c. Start the UI (terminal 2)

```bash
cd doc-intelligence/frontend
npm install
npm run build
npm start
```

Open **http://localhost:3200**:

1. **Documents** — drag in PDFs / Word / text files. They become searchable within seconds; the summary,
   topic and entities appear a bit later.
2. **Ask** — type a question. *Extractive* (default) quotes the exact answer and highlights it in the source;
   *Generative* lets the local LLM write an answer citing [1], [2]. Click a source to open the page.
3. **Search** (keyword / meaning / hybrid) and **Metrics**.

Want sample documents? `../.venv/Scripts/python training/build_eval_corpus.py` writes 35 Wikipedia articles
as PDFs into `data/eval/pdfs/`.

### 4d. Optional: reproduce the evaluation (~1 h on a laptop GPU)

```bash
cd doc-intelligence
../.venv/Scripts/python training/download_assets.py
../.venv/Scripts/python training/build_eval_corpus.py
../.venv/Scripts/python training/eval_retrieval.py
../.venv/Scripts/python training/eval_qa.py
../.venv/Scripts/python training/eval_summarization.py
../.venv/Scripts/python training/eval_generation.py --variants prompt_v1,prompt_v2,prompt_v2+reader_gate
../.venv/Scripts/python scripts/make_results.py
```

### 4e. Tests

```bash
cd doc-intelligence/backend
../../.venv/Scripts/python -m pytest
```

20 tests with tiny fake models — no downloads, no GPU.

---

## 5. Docker (alternative to sections 3–4)

Each project has a `docker-compose.yml` (API + PostgreSQL + UI): `docker compose up --build` inside the project
folder. Create the models first (3a / 4a), because the images copy them. On Windows, Docker Desktop needs
WSL 2 (`wsl --install` in an administrator terminal, then restart). The Docker setup has been validated
(`docker compose config`, clean-environment tests, CI jobs) but not yet built on the development laptop.

## 6. Troubleshooting

| symptom | fix |
|---|---|
| A model download sits at 0 bytes | `export HF_HUB_DISABLE_XET=1`, restart the command |
| Vision API says it runs on CPU although you have a GPU | `pip install "onnxruntime-gpu[cuda,cudnn]"` (ONNX Runtime needs its own CUDA libraries) |
| Training crashes with "paging file is too small" | lower `workers` in `training/configs/experiments.yaml`, close other heavy programs |
| `Port 3100/3200/8100/8200 already in use` | another copy is running — stop it, or pass `--port` to `scripts/dev_api.py` |
| Webcam page shows "camera blocked" | open the dashboard via `http://localhost:3100` (browsers only allow cameras on localhost/https) |
| Generative answers are very slow | no GPU: set `DI_ENABLE_GENERATOR=false`; extractive mode stays fast |
