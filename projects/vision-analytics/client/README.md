# Vision Analytics — Client (React)

A React 19 dashboard built with **Vite**, written in **plain JavaScript** (`.jsx` / `.js`), styled with
Tailwind v4. It talks only to the Express server in `../server` — never to the Python ML service.
It runs detection on images, videos and a live webcam, shows tracking overlays, and charts the analytics
the server stores in MongoDB.

## Run it

```bash
npm install
npm run dev          # http://localhost:3100 — needs the server on :4100 (and the ML service + MongoDB behind it)
npm run build        # production build -> dist/ (the Express server serves it at http://localhost:4100)
npm run format       # Prettier
```

`vite.config.js` forwards every `/api` request — HTTP **and** the `/api/stream` WebSocket — to the Express
server (`API_PROXY_TARGET`, default `http://127.0.0.1:4100`). So the app always uses relative URLs like
`/api/runs`: no CORS, no hard-coded host. `VITE_API_URL` exists only for the rare case where the API is on a
different host than the page.

## Folder structure

```
index.html               the single HTML page; React mounts into <div id="root">
vite.config.js           React + Tailwind plugins, /api proxy to Express
src/
  main.jsx               createRoot + <BrowserRouter>
  App.jsx                shared shell (<Nav/> + <main>) and the URL -> page table (React Router)
  index.css              Tailwind + theme tokens (CSS variables, light/dark via prefers-color-scheme)
  pages/                 one file per URL
    Overview.jsx         /          totals + recent runs
    Live.jsx             /live      webcam over WebSocket
    Image.jsx            /image     single-image detection
    Video.jsx            /video     upload + progress polling
    RunPage.jsx          /runs/:id  checks the id -> components/RunDetail.jsx
    History.jsx          /history   filters synced to the URL (?kind=video&page=2)
    Compare.jsx          /compare   two models side by side
    Metrics.jsx          /metrics   offline evaluation results
    NotFound.jsx         any other URL
  components/            reusable UI (no page logic)
    ui.jsx               Card, StatCard, ErrorState, Slider, ProgressBar, TimingBreakdown…
    charts.jsx           BarList (HTML bars) and LineChart (hand-written SVG)
    DetectionImage.jsx   <img> + canvas overlay scaled to the displayed size
    VideoOverlayPlayer.jsx <video> + canvas overlay synced to currentTime
    LineEditor.jsx       click two points -> normalised counting line
    ErrorBoundary.jsx    shows "Something went wrong" instead of a blank page
    ModelSelect.jsx, FileDrop.jsx, ConfirmDialog.jsx, RunsTable.jsx, RunDetail.jsx, Nav.jsx
  lib/
    api.js               the only file that calls the server (one function per endpoint)
    draw.js              drawDetections(ctx, items, scaleX, scaleY) used by EVERY overlay
    hooks.js             useAsync (fetch + abort + reload), useDebounced, useElementSize, usePageTitle…
    format.js            number / date formatting, shortId (last 6 chars of a MongoDB id)
```

**Design choices worth knowing**

- **All data is fetched in the browser** with `useAsync` (loading / empty / error states on every request,
  and an `AbortController` so a slow old response never overwrites a newer one).
- **One API client** (`lib/api.js`). Pages never call `fetch` directly; errors become an `ApiError`
  carrying the server's `detail` message so the UI can show something useful.
- **One drawing function.** Boxes come back in *original image pixels*; canvases are the *displayed* size.
  So every overlay calls `drawDetections(ctx, items, displayW / imageW, displayH / imageH)`. Colours come
  from a hash of the label, so "car" is the same colour in every overlay and chart.

## How each page talks to the API

| Page | Requests |
|---|---|
| `/` Overview | `GET /api/analytics/overview`, `GET /api/health`, `GET /api/runs?limit=8` (independent, so one failure doesn't blank the page) |
| `/image` | `POST /api/detect/image` (multipart: file, model, conf, iou, `save=false`) on every change, debounced 350 ms. **Save to history** repeats it with `save=true` and links the returned `run_id`. |
| `/live` | WebSocket `/api/stream` — see below |
| `/video` | `POST /api/detect/video` (file, model, conf, `line` as a JSON string) → `202` with a queued run → poll `GET /api/runs/{id}` every 1 s → go to `/runs/{id}` when `done` |
| `/runs/:id` | `GET /api/runs/{id}` (polled while queued/running). Video: `/video` + `/frames` for playback overlay. Video & live: `/timeseries`, `/events`. Image: `/detections`. Buttons: `/export.csv`, `DELETE /api/runs/{id}` (confirm dialog). |
| `/history` | `GET /api/runs?kind=&label=&q=&limit=20&offset=` — filters & page live in the URL, so views are shareable and Back works |
| `/compare` | `POST /api/compare` with `models="a,b"` |
| `/metrics` | `GET /api/metrics`; each section (`coco_baseline`, `domain_gap`, `finetuned`, `models`) shows "not run yet" if missing |

### The live WebSocket protocol (`src/pages/Live.jsx`)

1. `getUserMedia` → `<video>`. Each frame is drawn onto a hidden canvas at ≤640 px wide and
   encoded with `canvas.toBlob('image/jpeg', 0.7)`.
2. Open the socket (`/api/stream` on the Express server, which relays it to the Python ML service), send
   `{"type":"config", model, conf, line, width, height}`, wait for `{"type":"ready"}`.
3. Send **one** frame (binary), wait for its `{"type":"result"}`, then send the next.
   Only one frame is ever in flight — this is *back-pressure*: if the server is slow the frame
   rate drops instead of a queue of stale frames building up (which would make boxes lag).
4. Each result's `tracks` (box, `#id`, score, trail) are drawn on a canvas over the video,
   scaled from the *sent* frame size to the displayed size. `events` are accumulated into
   cumulative in/out counts; `occupancy`, `timing` and `server_fps` feed the side panel.
5. **Stop** sends `{"type":"stop"}`; the server saves the session in MongoDB and replies
   `{"type":"saved", run_id}` → link to `/runs/{run_id}`.

The counting line is fixed for a session (the ML service builds the line counter from the config
message), so line, model and confidence are locked while analysing — stop and start again to change them.

### Video overlay sync (`src/components/VideoOverlayPlayer.jsx`)

`/frames` returns `{fps, width, height, frames: [{t, tracks}]}` — the tracker output for every
frame. A `requestAnimationFrame` loop reads `video.currentTime`, computes
`index = floor(currentTime * fps)` and redraws only when the index changes, so the boxes stay
in sync while playing, pausing and seeking.
