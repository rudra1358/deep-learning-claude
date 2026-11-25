# Document Intelligence: web UI

A Next.js 15 (App Router) + React 19 + TypeScript + Tailwind v4 front end for the document-intelligence
API in `../backend`. You can upload documents, ask questions and get cited answers, search the documents,
and view the evaluation metrics.

## Run it

```bash
# 1. start the backend first (in another terminal)
cd ../backend && uvicorn app.main:app --port 8200

# 2. start the UI
cd frontend
npm install
npm run dev            # http://localhost:3200
```

The UI calls the API at `NEXT_PUBLIC_API_URL`, which defaults to `http://localhost:8200`. To change it,
copy `.env.example` to `.env.local` and edit it. The backend's CORS list already allows
`http://localhost:3200`.

| Script | What it does |
|---|---|
| `npm run dev` | dev server with hot reload on port 3200 |
| `npm run build` / `npm start` | production build / serve it on port 3200 |
| `npm run typecheck` | `tsc --noEmit`, strict TypeScript |
| `npm run lint` | ESLint (Next.js rules; `any` is an error) |

Docker: `docker build -t doc-intel-frontend --build-arg NEXT_PUBLIC_API_URL=http://localhost:8200 .`
then `docker run -p 3200:3200 doc-intel-frontend`.

## The big picture (say this first in an interview)

1. **The browser talks to the API directly.** Every page is a Client Component (`"use client"`). It
   fetches data in `useEffect` or in a button handler, keeps it in `useState`, and renders it. There
   is no Next.js API layer in between, which keeps the code simple. The trade-off is that the API
   needs CORS and its URL is public.
2. **All HTTP lives in one file, `lib/api.ts`.** It holds TypeScript types that mirror
   `backend/app/schemas.py`, plus one small function per endpoint. Pages never call `fetch`
   themselves, so an API change is fixed in one place.
3. **Every view has three states: loading, empty and error.** The shared components in
   `components/ui.tsx` draw them. If the backend is down, `fetch` throws, and `lib/api.ts` turns that
   into the message "Cannot reach the API at …".

```
browser ──fetch──> FastAPI :8200
  app/page.tsx            POST /api/ask, GET /api/chats, GET /api/chats/{id}, DELETE /api/chats/{id}
  app/documents/page.tsx  POST /api/documents (multipart), GET /api/documents (polled), DELETE /api/documents/{id}
  components/DocumentViewer.tsx  GET /api/documents/{id}, GET /api/documents/{id}/pages, link to /file
  app/search/page.tsx     POST /api/search
  app/metrics/page.tsx    GET /api/metrics
  components/HealthBadge  GET /api/health (every 15 s)
```

## File by file

### Config (project root)
- **package.json**: dependencies and scripts. The ports are pinned to 3200 so this app never collides
  with the other projects.
- **next.config.ts**: `output: "standalone"` makes `next build` emit a self-contained `server.js` with
  only the `node_modules` it needs. The Dockerfile copies just that output.
- **tsconfig.json**: `strict` plus `noUncheckedIndexedAccess`, so `array[i]` is typed as possibly
  `undefined`. That is why the code has a few `!` after lookups that we know are safe.
- **eslint.config.mjs**: Next.js recommended rules, with `no-explicit-any` set to error.
- **postcss.config.mjs**: turns on Tailwind v4.
- **Dockerfile / .dockerignore**: a 3-stage build (deps → build → runner) on `node:22-alpine`. It runs
  as a non-root user. `NEXT_PUBLIC_API_URL` is a *build arg* because Next.js inlines `NEXT_PUBLIC_*`
  values into the browser bundle at build time.

### `lib/`
- **api.ts**: the API contract.
  - The types (`DocumentOut`, `Source`, `AskOut`, `SessionDetail`, …) copy the Pydantic models field
    for field. `Ranks` values can be `null`: the engine returns `None` when a retriever did not find
    that chunk.
  - `request<T>()` is the one fetch helper. It (a) turns a network failure into a friendly `ApiError`
    with status 0, (b) reads FastAPI's `{"detail": ...}` on 4xx/5xx so the user sees *why* something
    failed (for example "409: document is still being processed"), and (c) handles 204 (no body) for
    DELETE.
  - `uploadDocuments` builds a `FormData` and appends every file under the same field name, `files`,
    which FastAPI reads as `list[UploadFile]`. The code deliberately does **not** set Content-Type: the
    browser adds `multipart/form-data; boundary=…` itself.
  - `fileUrl(id)` is a plain URL. The download is a normal `<a href>`, so the browser streams the file
    and no JavaScript is involved.
- **text.ts**: small pure helpers with no React.
  - `pageLabel`: "p. 3" or "pp. 3–4".
  - `viewerHref`: builds `/documents/{id}?page=&start=&len=&q=` for a source card.
  - `ranksLine`: builds the "why" line, e.g. "BM25 #3 · Dense #1 · before rerank #2".
  - `timingLine`: shows the stage timings.
  - `findIgnoringSpaces`: finds a chunk inside a page's text even when the whitespace differs. This
    is needed because the chunker re-joins sentences with single spaces while the page keeps its
    newlines. It collapses whitespace and records where each kept character came from, so the match
    maps back to positions in the original text.

### `app/` (one folder = one URL)
- **layout.tsx**: wraps every page in `<html>`, the nav bar and `<main>`. It is a Server Component
  because it needs no browser state.
- **globals.css**: colours are CSS variables, and a second set applies under
  `@media (prefers-color-scheme: dark)`, which is how dark mode works with no JavaScript. `@theme inline`
  exposes the variables to Tailwind, so classes like `bg-card` and `text-muted` work.
- **page.tsx, Ask (`/`)**: the chat page.
  - State: `turns` (the conversation), `sessionId`, `mode`, `docIds`, plus loading and error flags.
  - On submit, the question is shown at once (an optimistic update) before calling
    `ask({question, mode, session_id, document_ids})`. The `session_id` from the response is stored
    and sent with the next question, so the server continues the same conversation. If the call
    fails, the question is removed again and put back in the text box.
  - `historyVersion` is a counter. Bumping it makes the sidebar's `useEffect` reload `/api/chats`,
    so a new conversation appears in the list.
  - On mount it calls `/api/health`. If `generator_enabled` is false, the Generative option is
    disabled up front, so the user does not first hit a 503.
- **documents/page.tsx**: the upload zone plus the documents table.
  - Polling: `busy` is computed from the data ("is any doc queued, processing or enrich-running?").
    An effect starts `setInterval(load, 2000)` only while `busy` is true. When nothing is busy, the
    effect's cleanup clears the timer, so polling stops on its own.
  - Delete asks `window.confirm` first. A 409 from the server (document still processing) is shown
    in its own error box, so the next poll does not clear it.
- **documents/[id]/page.tsx**: a Server Component that only reads `params` and `searchParams` (both
  are Promises in Next 15) and passes plain numbers to `DocumentViewer`.
- **search/page.tsx**: a form with a query, mode select, rerank checkbox, top-k (kept between 1 and
  50, as the API requires) and the document filter. Each mode has a one-sentence explanation in the
  UI. Results reuse `SourceCard`, with the "why" ranks line added.
- **metrics/page.tsx**: one card per known section (retrieval, qa, summarization, ner, classifier,
  generation), each with a sentence on what it measures. A missing section shows
  "Not run yet - run training/<script>.py". Unknown extra sections are still rendered. QA gets one
  hand-picked headline table (oracle vs system, tuned threshold), because those numbers sit three
  levels deep in the JSON.

### `components/`
- **Nav.tsx**: the top links. `usePathname` highlights the current page and sets `aria-current`.
- **HealthBadge.tsx**: polls `/api/health` every 15 s and shows "API ok · cuda · N chunks ·
  generator on", or "API unreachable" in red. The `cancelled` flag stops a late response from
  updating a component that has already unmounted.
- **ui.tsx**: `Loading`, `Empty`, `ErrorBox` (with Retry) and `StatusBadge`.
- **ChatSidebar.tsx**: the chat history list, with load, select and delete (after a confirm). On
  phones it is collapsed behind a "History" button.
- **AssistantMessage.tsx**: renders one answer.
  - `Turn` is the UI's single shape for a message. Two converters produce it:
    `turnFromAsk(AskOut)` for a fresh answer and `turnFromHistory(MessageOut)` for a reloaded chat
    (where the details live in `meta`).
  - Not found: shows "Not found in your documents."
  - Extractive: shows the exact quote, the confidence, and the quote highlighted inside ±120
    characters of its source, using `highlight.start/end`.
  - Generative: shows the text with `[n]` chips (via `AnswerText`).
  - After the answer come the numbered source cards and the timings.
- **AnswerText.tsx**: a regex finds `[1]` or `[2, 3]`, and each number from 1 to `sources.length`
  becomes a link to `#m{turn}-src-{n}` (the source card's DOM id), so clicking scrolls to the
  evidence. Out-of-range numbers stay plain text.
- **SourceCard.tsx**: one retrieved chunk.
  - Shows the number, the filename (linked to the viewer), the page and the score.
  - The snippet is centred on the answer span when this card holds the extractive answer
    (`highlight.source === n`), and "Show more" reveals the full chunk.
  - Also exports `Highlighted`, which wraps `[start, end)` in `<mark>`.
- **DocFilter.tsx**: a dropdown of checkboxes over **ready** documents. An empty selection sends
  `document_ids: null`, which means "all documents".
- **UploadZone.tsx**: drag-and-drop or click-to-choose upload.
  - `onDragOver` calls `preventDefault()`, which is what allows a drop; `onDrop` reads
    `e.dataTransfer.files`.
  - Extensions are checked on the client for a friendlier error than the server's 415.
  - The input is reset after each upload, so the same file can be picked again.
- **DocumentViewer.tsx**: the viewer.
  - Loads the document and its pages in parallel with `Promise.all`.
  - Shows the topic label, entity chips grouped PER / ORG / LOC / MISC with counts, the summary, the
    Download button and every page.
  - To highlight the cited chunk on `?page=N`, it first searches the page for the chunk's first
    words (`q`), ignoring whitespace differences, and falls back to the `start` offset from the API.
    The highlight runs for `len` characters, clipped to the end of the page, because a chunk can
    continue onto the next page.
  - After rendering, an effect scrolls the `<mark>` into view.
- **MetricsView.tsx**: a generic JSON-to-tables renderer. Rules:
  - plain values → a key/value list
  - an object whose children are "objects of numbers" → a comparison table (a row per child, a
    column per metric)
  - a list of objects → a table
  - a list of lists of numbers → a confusion matrix, using a sibling `labels` list for headers
  - anything else → a sub-heading, then the same rules again (recursion)

  Score-like columns (recall, MRR, F1, EM, ROUGE, accuracy, …) get a small bar, and the best value in
  each is bold. Because nothing is hard-coded, new keys from the training scripts never crash the
  page.

## Accessibility and responsiveness
- Every input has a label (visible or `sr-only`), and status and error messages use `role="status"`
  or `role="alert"`.
- The answer-mode toggle is real radio inputs, so keyboard and screen readers work. A visible focus
  ring is defined globally.
- At 375 px the nav wraps, the chat sidebar collapses, and wide tables scroll inside their own box
  (`overflow-x-auto`) rather than widening the page.
