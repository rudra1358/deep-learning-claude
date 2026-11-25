# Document Intelligence: web UI (React + Vite)

React 19 + Vite + plain JavaScript + Tailwind v4 + React Router. You can upload documents, ask questions and
get cited answers, search the documents, and view the evaluation metrics.

## Run it

```bash
# start the ML service and the server first (see ../README.md)
npm install
npm run dev            # http://localhost:3200  (calls to /api are proxied to the server on :4200)
npm run build          # static files in dist/ - the Express server serves them on http://localhost:4200
```

The UI calls `/api/...` on its own origin. In development Vite forwards those calls to the server
(`vite.config.js`), so there is no CORS to configure. `VITE_API_URL` (see `.env.example`) points it at another
server instead.

| Script | What it does |
|---|---|
| `npm run dev` | dev server with hot reload on port 3200 |
| `npm run build` | production build into `dist/` |
| `npm run preview` | serve the built `dist/` on port 3200 (for a quick check) |

## The big picture (say this first in an interview)

1. **A single-page app.** `index.html` loads `src/main.jsx`, which mounts `<App/>`. React Router changes the
   page without reloading; the Express server sends `index.html` for any non-API URL so deep links work.
2. **All HTTP lives in one file, `src/lib/api.js`.** One small function per endpoint, plus a comment that
   documents every response shape. Pages never call `fetch` themselves, so an API change is fixed in one place.
3. **Every view has three states: loading, empty and error.** The shared components in `src/components/ui.jsx`
   draw them. If the server is down, `fetch` throws and `api.js` turns that into "Cannot reach the API…".
4. **State is plain React:** `useState` for data, `useEffect` to load it (with cleanup for timers), no
   global store — each page owns its data.

```
browser ──fetch /api──> Express :4200 ──> Python ML :8200
  pages/AskPage.jsx        POST /api/ask, GET /api/chats, GET /api/chats/:id, DELETE /api/chats/:id
  pages/DocumentsPage.jsx  POST /api/documents (multipart), GET /api/documents (polled), DELETE /api/documents/:id
  components/DocumentViewer.jsx  GET /api/documents/:id, GET /api/documents/:id/pages, link to /file
  pages/SearchPage.jsx     POST /api/search
  pages/MetricsPage.jsx    GET /api/metrics
  components/HealthBadge   GET /api/health (every 15 s; says whether MongoDB or the ML service is down)
```

## File by file

### Config
- **package.json** — React, React Router, and the build tools (Vite, Tailwind). Port 3200 so this app never
  collides with the other projects.
- **vite.config.js** — React + Tailwind plugins, dev port, and the `/api` proxy to the server.
- **index.html** — the one HTML page; `<div id="root">` is where React renders.

### src/
- **main.jsx** — mounts React with `BrowserRouter` (URL ↔ page) and `StrictMode` (extra dev-time checks).
- **App.jsx** — the shell (nav + main area), the route table and the browser-tab title per page.
- **index.css** — Tailwind import + colour variables for light/dark mode (the OS setting picks one).

### src/lib/
- **api.js** — the fetch helper (`ApiError` with status; status 0 = server unreachable), one function per endpoint.
- **text.js** — pure helpers: page labels, viewer links, "why was this retrieved" line, timing line,
  whitespace-insensitive text search (used to find a cited chunk inside a page).

### src/pages/
- **AskPage.jsx** — the chat: optimistic question bubble, answer + sources, Enter to send, mode switch
  (generative disabled if the server says the LLM is off), document filter, history sidebar.
- **DocumentsPage.jsx** — upload zone + table; polls every 2 s only while something is processing.
- **DocumentPage.jsx** — reads `/documents/:id?page=&start=&len=&q=` and renders the viewer.
- **SearchPage.jsx** — raw retrieval with mode / re-rank / top-k controls and the rank of each hit per retriever.
- **MetricsPage.jsx** — the evaluation results, one section per training script.

### src/components/
- **Nav.jsx** — links (`NavLink` marks the current page) + health badge.
- **HealthBadge.jsx** — status pill, re-checked every 15 s.
- **UploadZone.jsx** — drag-and-drop / click upload, client-side type check.
- **DocFilter.jsx** — "only these documents" checkbox list (ready documents only).
- **ChatSidebar.jsx** — past conversations; collapsible on phones.
- **AssistantMessage.jsx** — one answer: text, quoted highlighted span, low-confidence suggestion, sources, timings;
  converts both fresh answers and stored messages into one shape.
- **AnswerText.jsx** — turns `[2]` citations in LLM answers into clickable chips.
- **SourceCard.jsx** — one retrieved chunk with the answer span highlighted and "Show more".
- **DocumentViewer.jsx** — document info (topic, summary, entities) and all pages, scrolled to the cited chunk.
- **MetricsView.jsx** — renders any metrics JSON as tables (score bars, best value bold, confusion matrix).
- **ui.jsx** — `Loading`, `Empty`, `ErrorBox`, `StatusBadge`.
