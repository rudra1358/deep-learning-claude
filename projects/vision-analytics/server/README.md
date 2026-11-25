# Vision Analytics — Server (Node + Express + MongoDB)

The API the browser talks to. It stores every run in **MongoDB** and calls the Python **ML service**
(`../ml-service`) whenever a model is needed. In production it also serves the built React app.

```
React client ──/api, /api/stream──▶ this server ──HTTP /detect, /jobs · WS /stream──▶ ML service (Python)
                                        │
                                     MongoDB
```

## Run it

```bash
npm install
npm start            # http://localhost:4100   (npm run dev = restart on file changes)
npm test             # needs MongoDB on localhost; uses a fake ML service, no Python needed
```

Needs MongoDB (`mongodb://127.0.0.1:27017`, or set `MONGO_URL`) and the ML service on :8100
(`python ../scripts/dev_api.py`). Settings come from environment variables or a `.env` file here —
see `.env.example`.

## Files

```
src/
  index.js          start: connect MongoDB -> HTTP server -> WebSocket relay -> resume video jobs
  app.js            the Express app: routers under /api, the built client (production), error handler
  config.js         every setting (PORT, MONGO_URL, ML_URL, DATA_DIR, MAX_UPLOAD_MB, CLIENT_DIST, JOB_POLL_MS)
  db.js             mongoose.connect + "is the database up?" for /api/health
  errors.js         HttpError + the error middleware: every error is {"detail": "..."}
  ml.js             fetch wrapper for the ML service (passes its errors on; 503 if it is down)
  jobs.js           video jobs: start in Python, poll progress, copy the result into MongoDB
  stream.js         live webcam WebSocket relay; saves the session when the user presses Stop
  models/Run.js     the Mongoose schema: one document per run, child rows embedded as arrays
  routes/
    system.js       GET /api/health, /api/models, /api/metrics
    detect.js       POST /api/detect/image, /api/compare, /api/detect/video (multer uploads)
    runs.js         history, run details, frames, video, events, timeseries, CSV, analytics overview
tests/api.test.js   real MongoDB ("vision-test" database) + a fake ML service written in Express
```

## Endpoints

| Endpoint | What it does |
|---|---|
| `GET /api/health` | `{status, mongo, ml, models, providers, default_model}` |
| `GET /api/models`, `GET /api/metrics` | forwarded from the ML service |
| `POST /api/detect/image` | multipart `file, model?, conf, iou, save` → detections; saved as a run unless `save=false` |
| `POST /api/compare` | multipart `file, models="a,b", conf, iou` → one result per model (not saved) |
| `POST /api/detect/video` | multipart `file, model?, conf, line?` → **202** + queued run; processed in the background |
| `GET /api/runs?kind=&label=&q=&limit=&offset=` | `{total, items}`, newest first |
| `GET /api/runs/:id` · `DELETE /api/runs/:id` | one run · delete it (409 while processing) |
| `GET /api/runs/:id/detections` `/frames` `/video` `/events` `/timeseries` `/export.csv` | run details |
| `GET /api/analytics/overview` | totals across all runs (MongoDB aggregation pipelines) |
| `WS /api/stream` | live webcam, relayed to the ML service |

Errors are always `{"detail": "message"}` with a meaningful status: 404 not found / unknown model,
409 still processing, 413 too large, 415 wrong file type, 422 bad parameter, 503 ML service down.
