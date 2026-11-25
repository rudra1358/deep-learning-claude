# Vision Analytics — Frontend

A Next.js 15 (App Router) + React 19 + TypeScript + Tailwind v4 dashboard for the FastAPI
backend in `../backend`. It runs detection on images, videos and a live webcam, shows
tracking overlays, and charts the analytics the backend stores.

## Run it

```bash
npm install
cp .env.example .env.local        # optional: NEXT_PUBLIC_API_URL (default http://localhost:8100)
npm run dev                       # http://localhost:3100
npm run typecheck && npm run lint
npm run build                     # production build (output: "standalone")
```

`npm start` runs `next start -p 3100` for a quick local check (Next prints a harmless warning
because of `output: "standalone"`). In Docker the standalone server is used instead:

```bash
docker build --build-arg NEXT_PUBLIC_API_URL=http://localhost:8100 -t va-web .
docker run -p 3100:3100 va-web
```

`NEXT_PUBLIC_API_URL` is read **in the browser**, so it is baked in at build time and must be
the URL the *user's browser* can reach (not a Docker-internal hostname).

## Folder structure

```
app/                     one folder per route (App Router)
  layout.tsx             shared shell: <Nav/> + <main>
  page.tsx               /          Overview
  live/page.tsx          /live      webcam over WebSocket
  image/page.tsx         /image     single-image detection
  video/page.tsx         /video     upload + progress polling
  runs/[id]/page.tsx     /runs/:id  server wrapper -> components/RunDetail.tsx
  history/page.tsx       /history   filters synced to the URL
  compare/page.tsx       /compare   two models side by side
  metrics/page.tsx       /metrics   offline evaluation results
components/              reusable UI (no page logic)
  ui.tsx                 Card, StatCard, ErrorState, Slider, ProgressBar, TimingBreakdown…
  charts.tsx             BarList (HTML bars) and LineChart (hand-written SVG)
  DetectionImage.tsx     <img> + canvas overlay scaled to the displayed size
  VideoOverlayPlayer.tsx <video> + canvas overlay synced to currentTime
  LineEditor.tsx         click two points -> normalised counting line
  ModelSelect.tsx, FileDrop.tsx, ConfirmDialog.tsx, RunsTable.tsx, RunDetail.tsx, Nav.tsx
lib/
  api.ts                 typed API client — the TypeScript twin of backend/app/schemas.py
  draw.ts                drawDetections(ctx, items, scaleX, scaleY) used by EVERY overlay
  hooks.ts               useAsync (fetch + abort + reload), useDebounced, useElementSize…
  format.ts              number / date formatting helpers
```

**Design choices worth knowing**

- **All data is fetched in the browser** (client components). The API URL is a browser URL,
  and the pages are interactive anyway. Each request has loading / empty / error states.
- **One typed client** (`lib/api.ts`). Pages never call `fetch` directly; errors become an
  `ApiError` carrying FastAPI's `detail` message so the UI can show something useful.
- **One drawing function.** Boxes come back in *original image pixels*; canvases are the
  *displayed* size. So every overlay calls `drawDetections(ctx, items, displayW / imageW,
  displayH / imageH)`. Colours come from a hash of the label, so “car” is the same colour in
  every overlay and chart.
- **Theme** is CSS variables in `app/globals.css`, switched by `prefers-color-scheme`.

## How each page talks to the API

| Page | Requests |
|---|---|
| `/` Overview | `GET /api/analytics/overview`, `GET /api/health`, `GET /api/runs?limit=8` (independent, so one failure doesn't blank the page) |
| `/image` | `POST /api/detect/image` (multipart: file, model, conf, iou, `save=false`) on every change, debounced 350 ms. **Save to history** repeats it with `save=true` and links the returned `run_id`. |
| `/live` | WebSocket `ws://<api host>/api/stream` — see below |
| `/video` | `POST /api/detect/video` (file, model, conf, `line` as a JSON string) → `202` with a queued run → poll `GET /api/runs/{id}` every 1 s → redirect to `/runs/{id}` when `done` |
| `/runs/[id]` | `GET /api/runs/{id}` (polled while queued/running). Video: `/video` + `/frames` for playback overlay. Video & live: `/timeseries`, `/events`. Image: `/detections`. Buttons: `/export.csv`, `DELETE /api/runs/{id}` (confirm dialog). |
| `/history` | `GET /api/runs?kind=&label=&q=&limit=20&offset=` — filters & page live in the URL (`?kind=video&page=2`), so views are shareable and Back works |
| `/compare` | `POST /api/compare` with `models="a,b"` |
| `/metrics` | `GET /api/metrics`; each section (`coco_baseline`, `domain_gap`, `finetuned`, `models`) shows “not run yet” if missing |

### The live WebSocket protocol (`app/live/page.tsx`)

1. `getUserMedia` → `<video>`. Each frame is drawn onto a hidden canvas at ≤640 px wide and
   encoded with `canvas.toBlob('image/jpeg', 0.7)`.
2. Open the socket, send `{"type":"config", model, conf, line, width, height}`, wait for
   `{"type":"ready"}`.
3. Send **one** frame (binary), wait for its `{"type":"result"}`, then send the next.
   Only one frame is ever in flight — this is *back-pressure*: if the server is slow the frame
   rate drops instead of a queue of stale frames building up (which would make boxes lag).
4. Each result's `tracks` (box, `#id`, score, trail) are drawn on a canvas over the video,
   scaled from the *sent* frame size to the displayed size. `events` are accumulated into
   cumulative in/out counts; `occupancy`, `timing` and `server_fps` feed the side panel.
5. **Stop** sends `{"type":"stop"}`; the server saves the session and replies
   `{"type":"saved", run_id}` → link to `/runs/{run_id}`.

The counting line is fixed for a session (the server builds the line counter from the config
message), so line, model and confidence are locked while analysing — stop and start again to
change them.

### Video overlay sync (`components/VideoOverlayPlayer.tsx`)

`/frames` returns `{fps, width, height, frames: [{t, tracks}]}` — the tracker output for every
frame. A `requestAnimationFrame` loop reads `video.currentTime`, computes
`index = floor(currentTime * fps)` and redraws only when the index changes, so the boxes stay
in sync while playing, pausing and seeking.
