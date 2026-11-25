# Code walkthrough — every file, explained simply

This guide walks through the **code itself**, file by file, in the order the data flows. Keep the file open
next to this page while you read. (`HOW_IT_WAS_BUILT.md` explains the *big picture and decisions*;
`INTERVIEW_QA.md` has question-and-answer practice. This page is about *what each line is doing*.)

Each file section has the same four parts:

* **Job** — one sentence: why the file exists.
* **How it works** — the steps, in plain words.
* **Lines worth knowing** — the few lines an interviewer could point at.
* **Say it in an interview** — a one- or two-sentence answer you can memorise.

---

## 0. The map

```
                 ┌──────────── TRAINING (offline, PyTorch + Ultralytics) ────────────┐
 VisDrone/COCO → prepare_data.py → evaluate.py (baseline) → train.py → evaluate.py → export_onnx.py
                                                                                          │
                                                                              models/*.onnx  +  reports/metrics.json
                                                                                          │
                 ┌──────────── SERVING (online, NumPy + OpenCV + ONNX Runtime) ──────────┘
 image/frame → preprocess.py → detector.py (ONNX) → postprocess.py → tracker.py (+ kalman.py) → analytics.py
                                       ▲                                                          │
                     app/models_registry.py                                         app/pipeline.py
                                       ▲                                                          │
          ml-service/app/routers/detect.py · jobs.py · stream.py  ←── FastAPI (app/main.py, no database)
                                       ▲  HTTP + WebSocket (internal)
          server/ (Node + Express): routes/*.js · jobs.js · stream.js (relay)  ──→  MongoDB (models/Run.js)
                                       ▲  HTTP /api/* + WebSocket /api/stream
          client/ (React + Vite, plain JS): lib/api.js → pages/*.jsx → lib/draw.js (boxes on a <canvas>)
```

**Three tiers (a "MERN + Python" stack).** **M**ongoDB, **E**xpress, **R**eact and **N**ode build the web
app; Python does only the deep learning:

| tier | folder | language | its one job |
|---|---|---|---|
| client | `client/` | React (JavaScript) | the dashboard in the browser |
| server | `server/` | Node + Express | the API the browser talks to; owns MongoDB (runs, history, analytics) |
| ML service | `ml-service/` | Python + FastAPI | run the model, track, count - stores nothing |

The browser **never** calls Python directly. Why split it like this? Each part uses the language that is best
at it (Python for NumPy/ONNX, JavaScript for the web), each can be tested alone (the server tests use a fake
ML service; the ML tests use a fake model), and they scale separately (the ML service needs the GPU; the web
server does not).

**Two separate worlds.** Training needs PyTorch and Ultralytics (big, GPU). Serving needs only NumPy,
OpenCV and ONNX Runtime (small, runs anywhere). The `.onnx` file is the bridge between them. This split is
the single most important design idea in the project.

**Suggested reading order (≈2 hours):**
`preprocess.py → postprocess.py → detector.py → kalman.py → tracker.py → analytics.py → pipeline.py →
routers/detect.py → video_jobs.py → routers/stream.py → server/src/models/Run.js → server/src/jobs.js →
server/src/stream.js → client/src/lib/draw.js → client/src/pages/Live.jsx`.
Training scripts can be read afterwards.

### One frame's journey (memorise this)

1. The browser grabs a webcam frame, encodes it as JPEG and sends the bytes over a WebSocket to the
   Express server, which relays them unchanged to the Python ML service (`server/src/stream.js`).
2. `stream.py` decodes the JPEG into a NumPy array (H × W × 3, BGR colour order — OpenCV's default).
3. `preprocess.py` letterboxes it to 640 × 640 and turns it into a float tensor of shape (1, 3, 640, 640).
4. `detector.py` runs the ONNX model → raw output of shape (1, 4 + classes, 8400).
5. `postprocess.py` keeps confident candidates, removes duplicates with NMS and maps boxes back to the
   original image.
6. `tracker.py` matches these boxes to the objects it already knows → each box gets a stable ID.
7. `analytics.py` updates occupancy counts and checks whether anyone crossed the counting line.
8. The result is sent back as JSON (Python → Express relay → browser), and `draw.js` draws the boxes on a
   `<canvas>` over the video. When the user presses Stop, the ML service sends the whole session and the
   Express server saves it in MongoDB.

---

## Part A — Serving-time inference (`ml-service/inference/`)

This folder is pure maths: no web framework, no database. That is why it is easy to unit-test.

### A1. [`preprocess.py`](../ml-service/inference/preprocess.py) — make any image fit the network

**Job:** turn an image of any size into the exact 640 × 640 input the network was trained on.

**How it works**

1. `letterbox()` computes one scale factor `r = min(640/h, 640/w)`. Using the *smaller* ratio
   guarantees the whole image fits.
2. It resizes with that single factor, so the aspect ratio is kept (a car stays car-shaped).
3. It pads the leftover space with grey (114, 114, 114). Grey is used because training used grey too, so
   the model has seen it before and ignores it.
4. It returns a `LetterboxInfo` (ratio and padding) because postprocessing later has to *undo* this.
5. `to_input_tensor()` converts BGR → RGB, reorders HWC → CHW (channels first, as PyTorch expects),
   scales 0–255 → 0–1 and adds a batch dimension: final shape `(1, 3, 640, 640)`.

**Lines worth knowing**

```python
r = min(size / h, size / w)                          # one scale for both sides -> no distortion
chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
return np.ascontiguousarray(chw[None]), info         # [None] adds the batch dimension
```

The resize always uses `cv2.INTER_LINEAR` — the same interpolation Ultralytics' own letterbox uses — so
the network sees pixels exactly like it did during validation. (An earlier version used `INTER_AREA` for
shrinking; matching the reference exactly is what made the parity check reach 100 %.)

**Say it in an interview:** "I letterbox instead of stretching so object shapes match what the model saw in
training, and I keep the scale and padding so I can map the boxes back exactly."

### A2. [`postprocess.py`](../ml-service/inference/postprocess.py) — from 8,400 guesses to a few boxes

**Job:** turn the raw network output into final boxes in original-image pixels.

**What the raw output is:** YOLO predicts on three grids (80×80, 40×40 and 20×20 cells = 8,400 cells in
total). Each cell guesses one box: `cx, cy, w, h` plus one score per class. So the output is
`(1, 4 + C, 8400)`.

**How it works** — `decode_yolo()`:

1. `.T` flips the output to `(8400, 4 + C)`: one row per candidate box.
2. For each row, take the best class (`argmax`) and its score.
3. Drop rows below `conf_thres` (0.25 by default). Usually only a few hundred survive.
4. `xywh_to_xyxy()` converts centre/size format into corner format (x1, y1, x2, y2).
5. `batched_nms()` removes duplicates (many neighbouring cells detect the same car).
6. `scale_boxes()` undoes the letterbox: subtract the padding, divide by the ratio, then clip to the image.

**The three helper functions**

* `box_iou(a, b)` — **Intersection over Union** for every pair at once, using NumPy broadcasting
  (`a[:, None]` against `b[None, :]` gives an M × K table). 1.0 means identical boxes; 0 means no overlap.
* `nms()` — greedy **Non-Maximum Suppression**: sort by score, keep the best box, delete every box that
  overlaps it by more than `iou_thres`, then repeat with what is left.
* `batched_nms()` — the **offset trick**: shift each class's boxes by `class_id × (largest coordinate + 1)`,
  so boxes of different classes can never overlap. A single NMS call then behaves like a separate NMS per
  class. (This way a person on a bicycle keeps both boxes.)

**Lines worth knowing**

```python
x1 = np.maximum(a[:, None, 0], b[None, :, 0])        # broadcasting -> every pair at once
order = order[1:][ious <= iou_thres]                 # NMS: drop everything overlapping the winner
out[:, [0, 2]] = (out[:, [0, 2]] - info.pad_x) / info.ratio   # undo letterbox
```

**Say it in an interview:** "The head predicts a box for every grid cell, so I filter by confidence, run
class-wise NMS to remove duplicates, and invert the letterbox to get original-image coordinates. I wrote this
myself, and the export script checks that it matches Ultralytics."

### A3. [`detector.py`](../ml-service/inference/detector.py) — run the network

**Job:** load an `.onnx` file once and run it on images. This is the only place the neural network runs at
serving time.

**How it works**

1. `_providers()` picks the hardware. "Providers" are ONNX Runtime's backends: it uses CUDA if available
   and requested, otherwise CPU.
2. `__init__` creates an `InferenceSession` with all graph optimisations on (operator fusion, constant
   folding).
3. It checks which provider was *actually* used. ONNX Runtime silently falls back to CPU when CUDA DLLs
   are missing, so the code logs a warning, or raises an error if CUDA was explicitly demanded.
4. It reads the class names from the model's metadata. Ultralytics stores them there as text like
   `"{0: 'person', ...}"`, and `ast.literal_eval` safely turns that into a dict. (It is **not** `eval`,
   so no code can run.)
5. `warmup()` runs two dummy images. The first call is always slow (memory allocation, CUDA kernel
   selection), and we don't want a real user to pay for it.
6. `__call__()` = preprocess → `session.run` → `decode_yolo`, and it times each stage with
   `time.perf_counter()`.

**Say it in an interview:** "Serving uses ONNX Runtime instead of PyTorch, so the server image is small and
the same file runs on CPU or GPU. I time preprocess, inference and postprocess separately so I know where
the milliseconds go."

### A4. [`kalman.py`](../ml-service/inference/kalman.py) — guess where each object will be next

**Job:** smooth a box's position and predict where it will be in the next frame.

**Intuition:** a Kalman filter keeps two things for every object: a **best guess** (`mean`) and **how
unsure it is** (`cov`, the covariance). Every frame it does two steps:

* **predict** — move the guess by its velocity (`x' = F x`). Uncertainty grows (`+ Q`), because time
  passed without new information.
* **update** — a detection arrives. Blend the prediction with the measurement. The *Kalman gain* `K`
  decides how much to trust each one. If the prediction was very unsure, trust the detector more.

**State:** 8 numbers, `[cx, cy, a, h, vcx, vcy, va, vh]`: centre, aspect ratio, height, and their
velocities. The detector only measures the first 4. That is what `H = np.eye(4, 8)` means: "look at the
first 4 entries".

**Lines worth knowing**

```python
self.F[i, i + 4] = 1.0                       # position_next = position + velocity (1 frame)
K = np.linalg.solve(S, self.H @ cov).T       # Kalman gain; solve() is more stable than inverting S
new_mean = mean + K @ (z - self.H @ mean)    # correct the prediction by the "surprise" (z - prediction)
```

The noise values are multiplied by box height `h`. A big (near) object can move more pixels per frame
than a tiny (far) one, so the uncertainty scales with size.

**Say it in an interview:** "A constant-velocity Kalman filter predicts each track's next box. That makes
matching work even when an object moves quickly, and lets a track coast through a few frames without a
detection."

### A5. [`tracker.py`](../ml-service/inference/tracker.py) — give every object a stable ID

**Job:** link detections across frames so "car #7" stays #7.

**Main objects**

* `Track` — one object being followed: ID, class, Kalman `mean`/`cov`, state, and a short `history` of
  centre points (used to draw the trail).
* `TrackState` — the life cycle: `NEW` (seen once, not trusted yet) → `TRACKED` → `LOST` (missing, kept
  for `track_buffer` = 30 frames in case it reappears) → `REMOVED`.
* `TrackerConfig` — every threshold in one place, with a comment for each.

**`_match(cost, thresh)`** uses the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`). Given
a table of "cost of pairing track *i* with detection *j*", it finds the one-to-one pairing with the lowest
total cost. Pairs costing more than `thresh` are rejected, because "the best available" can still be a bad
match.

**`_cost()`** builds that table: `cost = 1 − IoU × score`. A big overlap with a confident detection means a
low cost. Pairs with different classes get cost `1e6`, so a car can never "become" a person.

**`update()` — the 6 steps, one frame**

1. **Split detections** into *high* (≥ `high_thresh`) and *low* (between `low_thresh` and `high_thresh`).
2. **Predict:** run the Kalman `predict` for every track. (For tracks that aren't currently `TRACKED`, the
   height velocity is frozen so the box doesn't grow or shrink wildly while unseen.)
3. **Round 1:** confirmed + lost tracks vs. *high* detections → match → `_update_track()`.
4. **Round 2 (the ByteTrack idea):** tracks still unmatched vs. *low* detections. A half-hidden person often
   gets a score of 0.2. Most trackers would throw that box away; ByteTrack uses it to keep the ID alive.
   Tracks that are still unmatched become `LOST`.
5. **Tentative tracks** (seen once) try to match the leftover high detections. If they fail, they are
   removed. This is what stops one-frame false positives from becoming objects.
6. **Birth and bookkeeping:** leftover confident detections start `NEW` tracks. Tracks lost for more than 30
   frames are removed.

It returns only confirmed tracks that were matched **in this frame**.

**Say it in an interview:** "It's ByteTrack: Kalman prediction, then Hungarian matching on IoU cost in two
rounds, first high-confidence detections, then low-confidence ones to recover occluded objects. New tracks
need two frames to be confirmed, and lost tracks survive 30 frames."

### A6. [`analytics.py`](../ml-service/inference/analytics.py) — turn tracks into numbers people care about

**Job:** count line crossings, current occupancy, unique objects and dwell time.

**Line crossing, step by step**

* `_side(a, b, p)` is the 2-D **cross product**. Its sign says which side of line A→B the point P is on
  (positive = one side, negative = the other, 0 = on the line).
* `segments_intersect()` says two segments cross if each one's endpoints lie on opposite sides of the
  other (`d1 * d2 < 0 and d3 * d4 < 0`). Checking both directions matters: otherwise an object passing far
  beyond the end of a short line would still count.
* `LineCounter.update()` remembers each track's previous centre. If the segment *previous → current*
  crosses the counting line, that's an event. The sign of the previous side decides "in" or "out".
* `_counted` stores `(track_id, direction)`, so a person jittering on the line is counted **once** per
  direction.

**`SceneAnalytics`** records the first and last frame each track was seen. Dwell time =
`(last − first + 1) / fps` seconds.

**Say it in an interview:** "I count with a cross-product side test plus a segment-intersection check, and I
de-duplicate per track and direction, so jitter on the line doesn't inflate the count."

---

## Part B — The Python ML service (`ml-service/app/`)

This is a small FastAPI app that does **only** the deep-learning work. It has no database and no CORS: the
browser never calls it, only the Express server does (Part E).

### B1. [`config.py`](../ml-service/app/config.py) — every setting in one place

A `pydantic-settings` class. Each field has a default, and any field can be overridden with an
environment variable prefixed `VA_` (for example `VA_DEVICE=cpu`). `@lru_cache` means the settings are built
once and reused. This is the "12-factor app" idea: configuration lives in the environment, not in the code.
`VA_DATA_DIR` must be the same folder the Express server uses: the server saves video uploads there and this
service writes its outputs next to them.

### B2. [`main.py`](../ml-service/app/main.py) — the front door

* Creates the `FastAPI` app and plugs in the three routers (`detect`, `jobs`, `stream`).
* **`lifespan`:** code that runs once at startup. It loads and warms up the default model *before* the first
  request arrives.
* Three small endpoints: `/health`, `/models` and `/metrics` (the last one just returns `metrics.json`
  from training).

### B3. [`models_registry.py`](../ml-service/app/models_registry.py) — load each model once

Loading a model takes up to a second and uses GPU memory, so it must **never** happen per request.
`ModelRegistry.get(name)` loads the model on first use and caches it in a dict.

The `with self._lock` block is **double-checked locking**: two requests arriving at the same moment won't
both load the same model. (It checks, takes the lock, then checks again.) ONNX Runtime sessions are safe to
call from many threads, so one cached session serves everyone.

### B4. [`pipeline.py`](../ml-service/app/pipeline.py) — detector + tracker + analytics, glued together

**Job:** one object per video or webcam session that holds that stream's state (its own tracker and
counters). The detector is shared between streams; the state is not.

Two details worth knowing:

* `self.det_conf = min(conf, 0.1)`: the detector is run with a **low** threshold on purpose, because
  ByteTrack's second round needs the low-score boxes. The user's threshold is applied in the tracker
  instead (`high_thresh`).
* `process(frame)` returns a JSON-ready dict: tracks (id, label, score, box, last 15 trail points),
  occupancy, crossing events and timing.

`peak_counts()` is a tiny helper: from many `{label: count}` snapshots it keeps the highest count per label.
For example, "at most 5 cars were visible at once".

### B5. [`routers/detect.py`](../ml-service/app/routers/detect.py) — images and model comparison

* `POST /detect/image` — validate the file type and size → `cv2.imdecode` → detector → return boxes,
  counts and timing. Nothing is stored here; the Express server saves the result.
* `POST /compare` — the same image through 1–4 models. Each model gets one untimed warm-up call first
  so the timing comparison is fair.

Errors use proper HTTP codes: 404 unknown model, 413 too large, 415 wrong type, 422 invalid parameter.
The Express server passes these codes and messages straight on to the browser.

### B6. [`routers/jobs.py`](../ml-service/app/routers/jobs.py) + [`video_jobs.py`](../ml-service/app/video_jobs.py) — background video worker

* `POST /jobs/video {run_id, video_path, model, conf, line}` — the Express server calls this after it saved
  the upload. It checks the model exists and that `video_path` is **inside** `data/uploads` (so nobody can
  make the service read other files), queues the job and returns **202 Accepted** immediately.
* `ThreadPoolExecutor(max_workers=1)` is a queue with one worker thread, so videos run one at a time and
  don't fight over the GPU.
* `_process_run()` reads frames with `cv2.VideoCapture` → `pipe.process(frame)` → stores per-frame tracks.
  It updates the job's `progress` every 25 frames.
* It writes two files to `data/runs/<run_id>/` (too big for a database document):
  * `frames.json` — the tracks for every frame, so the browser can draw boxes over the video while it plays.
  * `playback.webm` — a re-encoded copy, because browsers can't play many upload formats (AVI, for example).
* and keeps a `result` in memory: tracks, crossing events, per-second counts and the summary.
* `GET /jobs/{run_id}` returns status, progress and (when done) the result. The Express server polls it and
  copies everything into MongoDB, then calls `DELETE /jobs/{run_id}`.
* The `try/except` in `_process()` makes sure a crash is recorded as `status = "failed"` with the error
  message, so the run never gets stuck on "running" forever.

**Say it in an interview:** "Long work never runs inside the HTTP request. The upload returns 202, a worker
in the Python service processes the video, and the Node server polls it and saves the result in MongoDB.
Job state in Python is only in memory, because MongoDB is the system of record. At scale I'd swap the thread
pool for a real queue (Celery/RQ with Redis, or BullMQ on the Node side)."

### B7. [`routers/stream.py`](../ml-service/app/routers/stream.py) — live webcam over a WebSocket

**The protocol** (also written at the top of the file):

1. The client sends `{"type": "config", model, conf, line, width, height}`. The service builds a `Pipeline`
   and answers `ready`.
2. The client sends **one** JPEG frame as binary. The service answers `result`. Only then does the client
   send the next frame.
3. The client sends `stop`. The service answers `session` with everything about the session (tracks, events,
   per-second counts, summary). The Express relay saves it in MongoDB and tells the browser `saved` + run id.

**Back-pressure** is the key idea in step 2. Because the client waits for each answer, a slow server just
lowers the frame rate. Frames never pile up in a queue, which would make the boxes lag further and further
behind the video.

`run_in_threadpool(pipe.process, frame)` runs the heavy inference in a worker thread, so the async event
loop stays free to serve other connections.

### B8. [`schemas.py`](../ml-service/app/schemas.py) — the contract with the server

Pydantic models describing what the endpoints accept and return. FastAPI uses them to validate requests
(for example `run_id` may only contain letters, digits, `_` and `-`) and to generate the interactive docs
at `http://localhost:8100/docs`.

---

## Part C — Training (`training/`)

Run order: `run_all.ps1`. Every script is a step you can run on its own.

### C1. [`common.py`](../training/common.py)
Shared folder paths, the 10 VisDrone class names, the VisDrone→COCO class mapping (used to measure the
"domain gap"), `seed_everything()` for reproducibility, and `update_metrics()`. That last function merges
results into `metrics.json`, which the API serves to the dashboard.

### C2. [`prepare_data.py`](../training/prepare_data.py) — Step 1: data + EDA
* Downloads VisDrone using Ultralytics' official recipe, pointed at our `datasets/` folder. It converts the
  labels to **YOLO format**: one `.txt` per image, one line per object:
  `class cx cy w h` (all 0–1).
* `eda()` produces the exploratory data analysis: class counts (shows the imbalance), objects per image,
  box sizes (shows that most objects are *tiny*) and image resolutions. Output:
  `reports/eda_visdrone.png` and `eda_visdrone.json`.
* `prepare_coco_subset()` makes a fixed, seeded 500-image COCO subset for fast baseline evaluation.

### C3. [`evaluate.py`](../training/evaluate.py) — Steps 2 and 4: measure
* `coco-baseline` — pretrained YOLO11n, YOLO11s and a Faster R-CNN on the same 500 COCO images, all scored
  by the same evaluator (`pycocotools`). This gives a fair accuracy-vs-speed comparison.
* `domain-gap` — COCO models on drone images **before** fine-tuning. Measured: YOLO11n gets only
  0.078 mAP50-95 (vs 0.406 on COCO). This number is the justification for fine-tuning.
* `finetuned` — after training: overall mAP, per-class AP, accuracy by object size and a confusion matrix,
  plus the same shared-class score as `domain-gap` so before/after is apples-to-apples
  (YOLO11n 7.8 → 19.4, YOLO11s 12.2 → 23.5 — see [RESULTS.md](RESULTS.md)).

Predictions are collected with `conf=0.001`, a deliberately low threshold, because mAP needs the full
precision–recall curve.

### C4. [`train.py`](../training/train.py) + [`configs/experiments.yaml`](../training/configs/experiments.yaml) — Step 3: fine-tune
* All hyperparameters live in the YAML file, not on the command line, so any run can be reproduced with one
  command: `python train.py --exp yolo11n-visdrone`.
* `load_experiment()` merges the `common` settings with the experiment's own settings.
* `model.train(...)` is Ultralytics' training loop. It starts from COCO weights (transfer learning), and
  only the final class layer is new.
* The comments in the YAML file record real constraints: batch size 12/6 to fit 6 GB of VRAM, and
  `workers: 2` because more workers exhausted Windows memory.

### C5. [`export_onnx.py`](../training/export_onnx.py) — Step 5: export and *prove* it's correct
1. `export()` — `.pt` → `.onnx` (opset 17, fixed 640 × 640 input).
2. `parity()` — **the important part.** Two checks per image:
   * *raw*: the **same input tensor** goes through PyTorch (`ref.model`) and ONNX Runtime; the raw
     `(1, 4+C, 8400)` outputs are compared number by number → proves the exported graph is the same function
     (measured: ≤ 0.0075 px box difference, ≤ 3e-5 score difference).
   * *end-to-end*: Ultralytics' `predict()` vs *our own* serving code (letterbox → ONNX → decode → NMS →
     rescale), boxes matched by IoU > 0.9. We hand Ultralytics *our* square letterboxed image, because its
     `predict()` normally pads to a minimal rectangle (`auto=True`) while our static ONNX model needs a
     640 × 640 square — without this the two sides see different pixels (that's why a first version only
     matched 92 % of boxes). Measured now: 100 % of 1,857 boxes on 4 models.
   A wrong padding, an RGB/BGR swap or a scaling bug would show up immediately as a low match rate.
3. `latency()` — CPU FP32, GPU FP32, CPU INT8 (dynamic quantisation) and GPU FP16. INT8 is measured but not
   served, because on many CPUs it was *slower* for convolutions. That's a result worth mentioning in an
   interview.

---

## Part D — The client (`client/`, React + Vite, plain JavaScript)

You don't need to know every line here. Know these four files well; the rest is ordinary React layout.
`src/main.jsx` mounts React, `src/App.jsx` maps URLs to pages with React Router (`/runs/:id` → `RunPage.jsx`),
and `vite.config.js` makes the dev server forward `/api` (HTTP **and** the WebSocket) to Express.

### D1. [`lib/api.js`](../client/src/lib/api.js) — the only file that talks to the server
* All URLs are relative (`/api/...`): in development Vite forwards them to Express, in production Express
  serves this app itself. So there is no CORS setup and no hard-coded host.
* `request()` is one wrapper around `fetch`. It turns "server not running" and HTTP errors into a
  readable `ApiError` message (the server always answers errors as `{"detail": "..."}`), and handles
  `204 No Content`.
* Each endpoint is a one-line function (`getRun`, `listRuns`, `detectImage` …). Pages never build URLs
  themselves. The comment at the top lists the main JSON shapes (run ids are MongoDB ObjectId strings).

### D2. [`lib/draw.js`](../client/src/lib/draw.js) — drawing boxes
* Boxes arrive in *original image pixels*, but the image is displayed smaller. So every coordinate is
  multiplied by `scaleX = displayedWidth / imageWidth` (and the same for Y).
* `fitCanvas()` multiplies the canvas size by `devicePixelRatio` so boxes stay sharp on high-DPI screens.
* `colorForLabel()` hashes the label text into a colour, so "car" is always the same colour on every page.

### D3. [`pages/Live.jsx`](../client/src/pages/Live.jsx) — the webcam page (the most involved file)
1. `getUserMedia` opens the camera (browsers allow this only on `https://` or `localhost`).
2. A WebSocket opens to `/api/stream` (the Express relay) and sends the `config` message.
3. `sendFrame()` draws the current video frame onto a hidden canvas, compresses it with
   `canvas.toBlob(…, "image/jpeg")` and sends the bytes.
4. `ws.onmessage` receives a `result`, saves it to state (which redraws the overlay) and **only then** calls
   `sendFrame()` again. This is the client half of back-pressure.
5. Stop sends `{"type":"stop"}` and shows a link to the saved run.

Refs (`useRef`) hold things that must not trigger a re-render (the socket, the stream, the current phase).
State (`useState`) holds things the screen shows.

### D4. [`components/VideoOverlayPlayer.jsx`](../client/src/components/VideoOverlayPlayer.jsx) — boxes over a recorded video
A `requestAnimationFrame` loop reads `video.currentTime`, computes the frame index
`floor(currentTime × fps)`, looks up that frame in `frames.json` and redraws only when the index changes.
Seeking, pausing and changing speed all work for free, because the video element is the clock.

The other files (`History.jsx`, `Compare.jsx`, `Metrics.jsx`, `charts.jsx`, `ui.jsx` …) are plain React:
fetch with `useAsync` from `lib/hooks.js`, then render cards, tables and SVG charts.

---

## Part E — The server (`server/`, Node + Express + MongoDB)

The browser talks only to this server. It owns the database and calls the ML service when it needs a model.

### E1. [`src/index.js`](../server/src/index.js) and [`src/app.js`](../server/src/app.js) — start-up and routes
`index.js` connects to MongoDB, starts the HTTP server, attaches the WebSocket relay and resumes watching
video jobs that were running when the server last stopped. `app.js` builds the Express app: JSON body
parser → the three routers under `/api` → in production the built React app (`client/dist`) with a
fallback to `index.html`, so a refresh on a page like `/runs/<id>` still works → the error handler. Tests
import `createApp()` directly without starting a real server.

### E2. [`src/models/Run.js`](../server/src/models/Run.js) — the MongoDB schema
One **document per run**. The SQL version needed five tables (runs, detections, tracks, events,
count_samples) joined by `run_id`. In MongoDB the child rows are **embedded arrays** inside the run, because
they are always read and deleted together with their run: one read, no joins, and deleting the run deletes
everything. The per-frame data (`frames.json`) is far bigger, so it stays on disk (a MongoDB document is at
most 16 MB). Indexes: `created_at` for the newest-first history, `detections.label` / `tracks.label` for
"runs that contain a car". The `toJSON` transform turns `_id` into `id` and hides the big arrays, which have
their own endpoints.

### E3. [`src/ml.js`](../server/src/ml.js) and [`src/errors.js`](../server/src/errors.js) — talking to Python, and errors
`mlRequest()` wraps `fetch` to the ML service with a timeout. If Python answers an error (404 unknown model,
415 wrong type …) the same status and message go to the browser; if Python is down, the browser gets **503**.
`errorHandler` is Express's error middleware: every error becomes `{"detail": "..."}` with the right status
(multer's file-too-large → 413, an invalid ObjectId → 404).

### E4. [`src/routes/detect.js`](../server/src/routes/detect.js) — uploads
* Images: `multer.memoryStorage()` (small) → forwarded to Python as multipart → result saved with
  `Run.create()` → `run_id` returned.
* Videos: `multer.diskStorage()` streams the upload straight to `data/uploads` (a 200 MB video never sits in
  RAM), creates a run `{status: "queued"}`, starts the job in Python and returns **202 Accepted**. If Python
  refuses (unknown model), the half-created run and the file are removed again.

### E5. [`src/jobs.js`](../server/src/jobs.js) — following a video job
A loop per job: every second `GET /jobs/<id>` from Python and copy `status` and `progress` into MongoDB (the
browser polls `GET /api/runs/<id>` and sees the progress bar move). When Python says `done`, the whole result
is written into the run and Python is told to forget the job. A 404 from Python means it restarted (its jobs
live in memory), so the run is marked `failed` instead of staying "running" forever.
**Why poll instead of a callback from Python?** The ML service then needs to know nothing about the server
or MongoDB — it stays a pure compute service that is easy to test alone.

### E6. [`src/routes/runs.js`](../server/src/routes/runs.js) — history, details, CSV, analytics
History with filters (kind, label, filename search with an escaped, case-insensitive regex, pagination),
run details, delete (refused with **409** while a run is processing), frames, video (`res.sendFile` supports
HTTP Range requests, so seeking works), events, timeseries and CSV export. `/api/analytics/overview` counts
inside MongoDB with **aggregation pipelines**: `$unwind` turns each array element into its own row and
`$group` is MongoDB's `GROUP BY`.

### E7. [`src/stream.js`](../server/src/stream.js) — the live WebSocket relay
For each browser connection it opens a second WebSocket to Python and passes every message through
unchanged (binary frames stay binary). Messages sent before the Python socket is open are buffered. The
one message it changes is Python's final `session`: it saves that run in MongoDB and sends the browser
`{"type":"saved","run_id":...}` instead. Back-pressure still works, because the relay never queues frames.

**Say it in an interview:** "The Node server is the system of record and the only public API; Python is an
internal compute service. They agree on a small contract: multipart for images, a job API for videos, and a
WebSocket protocol for live frames."

---

## Part F — Tests and running it

* [`ml-service/tests/conftest.py`](../ml-service/tests/conftest.py) builds a **fake ONNX model** that has the
  same input and output shapes as YOLO11 but always "detects" the same 3 boxes (one is a deliberate
  duplicate). This lets the tests run the *whole* serving path in seconds with no GPU and no trained
  weights, and the duplicate checks that NMS really removes it.
* [`test_inference_core.py`](../ml-service/tests/test_inference_core.py) checks the maths: IoU against known
  values, NMS, letterbox round-trip, a Kalman filter following a constant-velocity object, tracker IDs
  surviving a short occlusion, low-score boxes keeping a track alive, and line counting.
* [`test_api.py`](../ml-service/tests/test_api.py) checks the ML endpoints end to end, including a real video
  job (frames.json + a playable WebM) and the WebSocket protocol.
  Run: `cd ml-service && ../.venv/Scripts/python -m pytest -q` → **25 passed**.
* [`server/tests/api.test.js`](../server/tests/api.test.js) uses a real MongoDB (a throw-away `vision-test`
  database) and a **fake ML service** written in a few lines of Express. It checks storage, history filters,
  analytics, CSV, the video-job polling and the WebSocket relay. Run: `cd server && npm test` → **9 passed**.
* `docker-compose.yml` starts MongoDB + ML service + server (which also serves the built dashboard) with one
  command. The `gpu` profile swaps in the CUDA ML image.
* `.github/workflows/vision-analytics.yml` (at the repository root) runs all of this on every push.

---

## Five things to be able to draw on a whiteboard

1. **Letterbox:** an image → scaled to fit → grey padding → and the reverse mapping for the boxes.
2. **NMS:** three overlapping boxes → keep the highest score → delete the boxes with IoU > 0.45.
3. **ByteTrack rounds:** tracks × high detections → leftover tracks × low detections → new tracks.
4. **Line crossing:** line A→B, a point's previous and current position, opposite signs → crossed.
5. **Back-pressure loop:** send frame → wait → result → send the next frame.

If you can explain those five plus "why ONNX at serving time", you can answer almost any question about
this project.
