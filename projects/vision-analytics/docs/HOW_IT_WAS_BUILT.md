# How this project was built — explained simply

This guide walks through the project in the order it was built, explaining **what** each piece does,
**why** it was designed that way, and **what an interviewer may ask**. Read it top to bottom once; after
that use it as revision notes. Real measured numbers are in [RESULTS.md](RESULTS.md).

---

## 0. The 30-second pitch (say this first in an interview)

> "I built a real-time video analytics system. A YOLO11 detector, fine-tuned on drone/CCTV traffic data
> (VisDrone) — which took it from 12.2 to 23.5 mAP on that footage — finds objects in each frame. A ByteTrack-style tracker that I implemented myself — Kalman
> filter plus Hungarian matching — links detections across frames so each object gets a persistent ID.
> On top of that, an analytics layer counts objects crossing a virtual line, measures dwell time and
> occupancy over time. The model is exported to ONNX and served by a Python FastAPI ML service. The web
> app is a MERN stack: a React dashboard talks to a Node/Express API, which stores every run in MongoDB and
> calls the Python service over HTTP and WebSockets. The dashboard shows live overlays, history and
> model comparisons. The whole pipeline runs at about 37 FPS on a laptop GPU and 19 FPS on CPU, and it is
> packaged with Docker Compose."

Then be ready to go deep on any box in this diagram:

```
 Browser (React)            Node + Express server              Python ML service (FastAPI)
 ───────────────            ─────────────────────              ───────────────────────────
 webcam frames ──WS──▶  relay /api/stream ──WS──▶  decode ─▶ letterbox ─▶ ONNX Runtime (YOLO11)
 image / video ──HTTP─▶ multer upload ─────HTTP──▶       ─▶ decode + NMS ─▶ ByteTrack tracker
                         validate, save run                 ─▶ analytics (line crossings, dwell, occupancy)
 overlay + charts ◀──── JSON ◀──────────────────── results ◀┘   video jobs: background worker thread
                              │                                 writes frames.json + playback.webm
                              ▼
                        MongoDB (runs, history, analytics)
```

---

## 1. Problem framing

**Business problem:** stores, warehouses and traffic operators want numbers from cameras — how many cars
passed, how many people are in a zone, how long they stayed — without someone watching every feed.

**Translated into ML/engineering requirements:**

| Need | Technical requirement |
|---|---|
| "what is in the frame" | object **detection**: box + class + confidence |
| "how many *different* cars passed" | multi-object **tracking**: same object → same ID across frames |
| "cars going in vs out" | **line-crossing** logic on track trajectories |
| near real time | ≥ 15 FPS at 640 px on a mid-range GPU, CPU fallback |
| trustworthy | measured mAP on held-out data, every box comes from real inference |
| maintainable | config-driven training, tests, Docker, CI |

**Interview tip:** always start from the business question, then say how you turned it into measurable
requirements. That is what "industry grade" means.

---

## 2. Data (training/prepare_data.py)

**Datasets used**
* **COCO 2017** — 80 everyday classes. The pretrained YOLO weights were trained on it. We only use a
  fixed 500-image random subset of the *validation* set to measure the baseline.
* **VisDrone2019-DET** — ~6.5k training / 548 validation images from drones and elevated cameras, 10
  classes (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor). This is
  our *domain* data: top-down traffic scenes, very small and crowded objects.

**YOLO label format** — one `.txt` per image, one line per object: `class cx cy w h`, all divided by the
image width/height so they are between 0 and 1. Normalised coordinates make labels independent of image
resolution, so resizing never breaks them.

**EDA (exploratory data analysis)** — the script writes `training/reports/eda_visdrone.json` and a plot.
What we look for and why:
* **Class imbalance** — cars and pedestrians dominate; tricycles and buses are rare. Rare classes will
  have lower AP; this explains per-class results later.
* **Objects per image** — VisDrone has dozens to hundreds per image → we need `max_det=300` and a
  tracker that handles crowds.
* **Box sizes** — most boxes are *small* (< 32×32 px). Small objects are the main difficulty: after
  the backbone downsamples by 8/16/32, a 10-pixel pedestrian covers about one cell of the finest grid.

**Q: Why not just use COCO weights directly?** Because of **domain shift**: COCO photos are taken at eye
level with big objects; drone footage is top-down with tiny objects, and COCO has no "van" or
"tricycle". We *measure* this gap (section 4) instead of assuming it.

---

## 3. Model choice

**YOLO (You Only Look Once)** — a *one-stage* detector: a single forward pass predicts boxes and classes
for a dense grid of locations. YOLO11's head is **anchor-free**: each grid cell directly predicts the
distance to the box edges, so no hand-tuned anchor boxes are needed.

**Three output scales.** For a 640×640 input the head predicts on 80×80 (stride 8, small objects),
40×40 (stride 16) and 20×20 (stride 32, large objects) grids → 6400 + 1600 + 400 = **8400 candidate
boxes**. The raw ONNX output is therefore shape `(1, 4 + num_classes, 8400)`.

**Why YOLO11n and YOLO11s?** n (nano, ~2.6 M parameters) and s (small, ~9.4 M) are the two sizes
that run in real time on a laptop GPU. We train both and choose using the accuracy-vs-latency trade-off.

**Faster R-CNN baseline** — a *two-stage* detector: (1) a Region Proposal Network proposes ~1000
candidate regions, (2) a second head classifies and refines each one. Usually accurate but slower. We
evaluate it on the same COCO images to have a reference point for the speed/accuracy discussion.

| | one-stage (YOLO, SSD) | two-stage (Faster R-CNN) | set prediction (DETR) |
|---|---|---|---|
| idea | dense predictions in one pass | propose regions, then classify | transformer predicts a set of boxes, Hungarian loss |
| speed | fastest | slower | slow to train, no NMS needed |
| typical use | real-time video, edge | high-accuracy offline | research, end-to-end pipelines |

---

## 4. Evaluation metric: mAP (training/evaluate.py)

1. **IoU** (intersection over union) of two boxes = overlap area / union area. 1 = identical, 0 = no overlap.
2. A prediction is a **true positive** if it overlaps a not-yet-matched ground-truth box of the same
   class with IoU ≥ threshold; otherwise a **false positive**. Unmatched ground truths are **false negatives**.
3. Sort predictions by confidence and walk down the list: at each point compute
   precision = TP / (TP + FP) and recall = TP / (TP + FN). This gives a **precision–recall curve**.
4. **AP** = area under that curve (for one class). **mAP** = mean over classes.
5. **mAP@0.5** uses IoU threshold 0.5. **mAP@0.5:0.95** averages over thresholds 0.50, 0.55 … 0.95 —
   it rewards tight boxes and is COCO's main metric.

We evaluate with confidence threshold 0.001 (keep almost everything) because AP integrates over *all*
thresholds; the 0.25 used in the app is an operating point chosen afterwards.

**Experiments run** (numbers in RESULTS.md):
* *COCO baseline* — pretrained YOLO11n/s and Faster R-CNN on the same 500 COCO images, same evaluator
  (pycocotools), plus latency and parameter count.
* *Domain gap* — COCO-pretrained YOLO on VisDrone val, scored only on the classes both datasets share.
* *Fine-tuned* — our VisDrone models: mAP, precision, recall, per-class AP, AP by object size, confusion
  matrix — and the same shared-class score so before/after is an apples-to-apples comparison.

---

## 5. Training = transfer learning (training/train.py, configs/experiments.yaml)

* Start from COCO weights. The backbone already knows edges, textures and object parts. The 10-class
  detection head is new; everything else is *fine-tuned* (all layers keep training with a small
  learning rate).
* **Config-driven and seeded**: every hyperparameter lives in `experiments.yaml`, `seed=0`,
  `deterministic=true`. One command reproduces a run: `python training/train.py --exp yolo11n-visdrone`.
* **Augmentation**: mosaic (4 images stitched into one — more small objects and context variety per
  batch), random scale/translate, horizontal flip, HSV colour jitter. Mosaic is switched off for the last
  epochs (`close_mosaic`) so the model finishes on realistic images.
* **Cosine LR schedule** and **early stopping** (`patience=10`: stop if validation mAP has not improved
  for 10 epochs; keep `best.pt`, not `last.pt`).
* **Experiment tracking**: each run writes `results.csv` (per-epoch box/class/DFL losses, precision,
  recall, mAP) plus plots; a JSON summary goes to `training/reports/`.

**The three training losses (YOLO11):** box loss (CIoU — penalises low overlap, centre distance and
aspect-ratio mismatch), classification loss (binary cross-entropy per class), and DFL (distribution focal
loss — predicts each box edge as a small probability distribution, which gives sharper localisation).

---

## 6. Export to ONNX + parity check (training/export_onnx.py)

**ONNX** is a framework-neutral file format for neural network graphs. **ONNX Runtime** executes it on
CPU, CUDA, TensorRT… by switching "execution providers".

Why serve ONNX instead of PyTorch:
* the API container doesn't need PyTorch (~2 GB smaller image, faster startup);
* graph optimisations (operator fusion, constant folding) → faster CPU inference;
* same file on any hardware.

**The parity check is the most important step.** We run identical images through (a) Ultralytics/PyTorch
and (b) *our own* serving code (letterbox → ONNX Runtime → our decoding + NMS) and match the boxes. A bug
in our pre/post-processing (BGR/RGB swap, wrong padding offset, wrong scale) would show up as a low
match rate. This is how you prove "the deployed model is the model we evaluated". Measured: **100 % of
1,857 boxes matched on all four models**, and on identical input tensors the raw outputs of PyTorch and
ONNX Runtime differ by at most 0.0075 px (see problem 6 in section 11 for how that was reached).

**Optimisations measured:** FP16 on GPU (half-precision weights/activations — less memory bandwidth,
uses tensor cores) and INT8 dynamic quantisation on CPU (weights stored as 8-bit integers). Measured,
not assumed — and neither helped the way textbooks suggest: FP16 was *not* faster for the nano model at
batch 1 (19.0 → 21.6 ms; the model is too small to be compute-bound) and only ~10–25 % faster for the small
model; dynamic INT8 on CPU was *slower* than FP32 (55.6 → 72.6 ms). Static INT8 with calibration, OpenVINO
or TensorRT are the next things to try. Full table in RESULTS.md.

---

## 7. Serving-time pipeline (ml-service/inference/) — written by hand, no Ultralytics at runtime

### 7.1 Letterbox (preprocess.py)
Resize keeping the aspect ratio so the long side is 640, pad the rest with grey (114). Stretching would
distort shapes the model never saw in training. Keep `ratio`, `pad_x`, `pad_y` to map boxes back:
`x_original = (x_network − pad_x) / ratio`.

### 7.2 Decoding + NMS (postprocess.py)
1. Transpose `(1, 4+C, 8400)` → `(8400, 4+C)`; best class and score per candidate.
2. Drop candidates under the confidence threshold.
3. Centre format `(cx, cy, w, h)` → corners `(x1, y1, x2, y2)`.
4. **Non-max suppression**: the detector fires several overlapping boxes for one object. Greedy NMS:
   take the highest-scoring box, delete every other box with IoU > 0.45 with it, repeat.
5. **Class-wise NMS** with the *offset trick*: add `class_id × (max_coordinate + 1)` to every box so
   boxes of different classes can never overlap — then one NMS call handles all classes. (A person on a
   bicycle must keep both boxes.)
6. Undo the letterbox.

### 7.3 Tracking (kalman.py, tracker.py) — ByteTrack style
**Why:** detection works per frame and has no memory. Tracking answers "is this the same car as before?"

**Kalman filter** — state `[cx, cy, aspect, h]` + their velocities. Every frame:
*predict* (move the box by its velocity, uncertainty grows) → *update* with the matched detection
(blend prediction and measurement; the **Kalman gain** decides how much to trust each, based on their
uncertainties). Noise scales with box height, so near (big) objects may move more pixels per frame.

**Association** — cost matrix `1 − IoU(predicted track box, detection)`; the **Hungarian algorithm**
(`scipy.optimize.linear_sum_assignment`) finds the minimum-total-cost one-to-one matching in O(n³);
matches above a cost threshold are rejected. Matching across different classes is forbidden.

**ByteTrack's idea — two rounds:**
1. tracks ↔ *high*-confidence detections (≥ 0.5);
2. tracks still unmatched ↔ *low*-confidence detections (0.1–0.5). A partly occluded person often gets
   score 0.3; a normal detector pipeline throws it away and the track breaks. ByteTrack uses it to keep
   the track alive. (This is why the pipeline runs the detector at a low threshold internally and only
   *displays* boxes above the user's threshold.)

**Track lifecycle:** new (tentative) → confirmed after a 2nd match (kills one-frame false positives) →
lost (kept for 30 frames, can be re-found after an occlusion with the *same ID*) → removed.

### 7.4 Analytics (analytics.py)
* **Line crossing** — the sign of the 2-D cross product `(B−A) × (P−A)` tells which side of line AB a
  point P is on. When a track's centre changes sign between frames *and* the movement segment really
  intersects the line segment (not just its infinite extension), it crossed; direction = sign change.
  Each track counts once per direction → jitter on the line can't inflate counts.
* **Occupancy** per class per frame, **peak occupancy** per second (stored for charts), **dwell time**
  = frames visible / fps, **unique objects** = number of distinct track IDs.

---

## 8. Web API (server/, Node + Express) and ML service (ml-service/, Python)

The browser only talks to the Express server. The server stores everything in MongoDB and calls the Python
ML service whenever a model is needed.

| Endpoint (Express, public) | What it does | Python call behind it |
|---|---|---|
| `POST /api/detect/image` | upload image → detections, counts, timing breakdown; saved to history | `POST /detect/image` |
| `POST /api/compare` | same image through 2–4 models → side-by-side accuracy/latency | `POST /compare` |
| `POST /api/detect/video` | upload video (+ optional counting line) → **202 Accepted** + run id | `POST /jobs/video`, then polls `GET /jobs/{id}` |
| `GET /api/runs/{id}` | status / progress / summary (the client polls this) | — (MongoDB) |
| `GET /api/runs/{id}/frames` `/video` `/timeseries` `/events` `/export.csv` | playback overlay data, charts, CSV export | — (MongoDB + files) |
| `GET /api/runs?kind=&label=&q=` | searchable, paginated history | — (MongoDB) |
| `GET /api/analytics/overview` | totals across all runs (MongoDB aggregation pipelines) | — |
| `WS /api/stream` | live webcam: one JPEG frame in → tracks + analytics JSON out | relayed to `WS /stream` |
| `GET /api/models`, `/api/metrics`, `/api/health` | model registry, offline evaluation results, health of the whole stack | `/models`, `/metrics`, `/health` |

Design decisions worth explaining:
* **Why three tiers (React → Express + MongoDB → Python)?** Each language does what it is best at: Python for
  NumPy/OpenCV/ONNX Runtime, JavaScript for the web. The ML service is stateless (no database), so it can be
  tested with a fake model, restarted, or moved to a GPU machine without touching the web app; the server can
  be tested with a fake ML service. The browser never sees Python, so only one service is public.
* **Model loaded once** at startup and cached (`models_registry.py`); ONNX Runtime sessions are thread-safe.
  Loading per request would cost ~0.5 s and GPU memory each time.
* **Long work never blocks a request**: Express streams the video upload to disk (multer `diskStorage`, never
  the whole file in RAM), creates a `queued` run in MongoDB, starts the Python job and returns 202 immediately.
  A worker thread in Python processes it; the server polls its progress and, when it is done, copies the
  result into MongoDB. At larger scale the same contract maps onto a real queue (Celery/RQ + Redis, or BullMQ)
  with workers in their own containers.
* **WebSocket back-pressure**: the browser sends the next frame only after receiving the previous
  result. If the server is slow, the frame rate drops instead of a queue of stale frames building up
  (otherwise the overlay would lag further and further behind reality). The Express relay passes frames
  through one by one, so this still holds end to end.
* **CPU-bound inference runs in a thread pool** (`run_in_threadpool`) so the async event loop keeps
  serving other connections.
* **Validation everywhere**: file types, sizes, thresholds in (0, 1), line coordinates in [0, 1] →
  clear 4xx errors; failed videos are marked `failed` with the error message instead of crashing the worker.
* **Database**: MongoDB via Mongoose. One document per run with the detections, tracks, line-crossing events
  and per-second counts **embedded** as arrays (in SQL these were five tables joined by `run_id`): they are
  always read and deleted with their run, so one document = one read, no joins. Per-frame track positions for
  a video are a JSON file on disk (thousands of entries per minute of video; only read as a whole for
  playback, and MongoDB documents are limited to 16 MB).

---

## 9. Testing

* **Unit tests** (`tests/test_inference_core.py`): IoU known values, NMS, class-wise NMS, letterbox
  round-trip, YOLO decoding on a synthetic tensor, the Kalman filter learning a constant velocity,
  the tracker keeping IDs through a 5-frame occlusion, low-score recovery, no confirmation of one-frame
  noise, line-crossing direction and once-only counting.
* **ML service tests** (`ml-service/tests/test_api.py`) use a **generated fake ONNX model** with the exact
  YOLO input/output signature and known boxes — so the full path (upload → decode → ONNX Runtime → NMS →
  rescale → JSON) is tested deterministically, in CI, without GPU or real weights. Includes the background
  video job (and its `frames.json` / playable WebM), the WebSocket protocol, and error cases.
* **Server tests** (`server/tests/api.test.js`) use a real MongoDB and a **fake ML service** (a few lines of
  Express that answer like Python): storage, history filters, MongoDB aggregations, CSV export, deletion,
  video-job polling, the WebSocket relay and error cases.

---

## 10. Deployment

* `ml-service/Dockerfile` — slim Python image, only ONNX Runtime (no PyTorch), non-root user, health check.
* `ml-service/Dockerfile.gpu` — CUDA + cuDNN runtime with `onnxruntime-gpu`.
* `server/Dockerfile` — two stages: build the React client with Vite, then a small Node image that runs
  Express and serves the built client (one container, one port: 4100).
* `docker-compose.yml` — MongoDB + ML service + server; `docker compose up --build`.
* CI (`.github/workflows/ci.yml`) — Python lint + tests, server tests against a MongoDB service container,
  client format check + build, then the whole Docker stack with a smoke test, on every push.
* Configuration via environment variables (`VA_*` for Python; `MONGO_URL`, `ML_URL`, `DATA_DIR` for Node),
  12-factor style.

---

## 11. Real problems I hit while building it (use these for "tell me about a challenge")

Each one is a true story from building this project — describe it as *symptom → diagnosis → fix → lesson*.

1. **GPU silently not used.** *Symptom:* the "CUDA" benchmark was no faster than CPU. *Diagnosis:* ONNX Runtime
   quietly falls back to `CPUExecutionProvider` when it can't load CUDA libraries; its log said
   `cublasLt64_13.dll ... is missing` — the installed onnxruntime-gpu was built for **CUDA 13**, while the
   PyTorch wheels ship **CUDA 12** DLLs. *Fix:* install `onnxruntime-gpu[cuda,cudnn]` (pulls matching CUDA 13
   runtime wheels) and call `ort.preload_dlls()`. *Result:* 61 → 27 ms per frame, 15 → 33 FPS end-to-end.
   *Lesson / hardening:* `detector.py` now compares the provider it asked for with the one it got and logs a
   warning (or raises if CUDA was explicitly required) — silent fallbacks are the worst kind of bug.

2. **Training crashed: "The paging file is too small".** *Diagnosis:* on Windows, PyTorch DataLoader workers
   are separate *spawned* processes; each one re-imports torch and loads the CUDA DLLs (~1 GB of committed
   memory each), and Ultralytics' validation loader uses twice as many workers. 4 workers → 12 processes →
   the 16 GB laptop ran out of commit and the main process hung. *Fix:* `workers: 2`, documented in the config.
   *Lesson:* measure memory, not just VRAM; Linux `fork` hides this cost, Windows `spawn` doesn't.

3. **VRAM headroom.** A 1-epoch smoke test showed batch 16 peaking at 5.8 GB of the 6 GB card. On Windows
   (WDDM) an overflow spills into much slower shared system memory instead of failing loudly, so batch sizes
   were reduced (n: 12, s: 6) before the long run. *Lesson:* always do a short smoke run before an hours-long job.

4. **Uploaded videos wouldn't play in the browser.** OpenCV happily reads AVI/MPEG-4 Part 2, but browsers
   don't. OpenCV's H.264 encoder (openh264) wasn't usable in this build, but VP8 was — so the worker writes a
   `playback.webm` of exactly the frames it analysed, which also guarantees the overlay timing lines up
   frame-for-frame with `frames.json`.

5. **Wrong expectation in my own test.** The letterbox test initially failed: I had computed the padding for a
   1280×480 image as 160 px; the code said 200. Re-deriving by hand (480 × 0.5 = 240 px tall, (640 − 240) / 2 =
   200) showed the *test* was wrong, not the code. *Lesson:* when a test fails, check the expectation too.

6. **Parity check said only 92 % of boxes matched.** *Symptom:* the exported ONNX pipeline reproduced only
   92 % of Ultralytics' boxes, with score differences up to 0.23 — not good enough to claim "deployed model =
   evaluated model". *Diagnosis:* instead of assuming the export was broken, I split the check in two. Feeding
   the **same tensor** to PyTorch and ONNX Runtime gave raw differences below 0.008 px — so the export was
   exact and the difference was in the *inputs*. Reading Ultralytics' source showed its `predict()` letterboxes
   to a minimal rectangle (`auto=True`) with `INTER_LINEAR`, while our static model gets a 640×640 square
   (and I had used `INTER_AREA`). *Fix:* use `INTER_LINEAR`, and give the reference the same square input.
   *Result:* 100 % of 1,857 boxes matched on all four models. *Lesson:* when a comparison fails, first
   make sure both sides are given identical inputs — check the reference too, not just your own code.

7. **Text encoding on Windows.** `Path.read_text()` defaults to the legacy cp1252 code page on Windows and
   crashed on UTF-8 YAML; every file read/write now passes `encoding="utf-8"`.

---

## 12. Limitations and what I would do next (interviewers love this)

* **Small objects** remain the weakest point (see AP_small). Next: train at 960–1280 px or use tiled
  inference (SAHI) — at the cost of latency.
* Tracker is **motion + IoU only**; fast motion or long occlusions cause ID switches. Next: add an
  appearance embedding (re-identification model, as in DeepSORT / BoT-SORT).
* Tracker metrics (MOTA, IDF1, HOTA) were not measured because VisDrone-DET has no track IDs; next step:
  evaluate on VisDrone-MOT.
* Single worker thread for videos; scale out with a job queue and multiple GPU workers; batch frames for
  higher GPU throughput.
* TensorRT execution provider for a further GPU speed-up; INT8 *static* quantisation with calibration.
* Monitoring: track confidence distributions and class frequencies over time to detect **data drift**.
* Licence note: Ultralytics YOLO is AGPL-3.0 — commercial closed-source use needs their enterprise licence.

See [INTERVIEW_QA.md](INTERVIEW_QA.md) for practice questions with answers.
