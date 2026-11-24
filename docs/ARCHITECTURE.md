# Architecture

## Goals that drove the design

1. Depth of content first — so content is **typed data**, not free-form pages: the `Lesson` type
   (`web/lib/types.ts`) makes all 16 teaching sections mandatory at compile time.
2. Simulations must run the real algorithm — so each simulator's maths lives in a pure module
   (`web/lib/sim/*.ts`) that can be tested independently of React, and the component only renders it.
3. Fast, cheap, offline-capable — lessons are statically generated at build time; progress lives in
   localStorage; Python runs in the learner's browser (Pyodide). The backend is optional.

## Frontend (web/)

| Concern | Choice | Why |
|---|---|---|
| Framework | Next.js 15 App Router, React 19, TypeScript strict | Static generation of ~90 topic pages, code splitting per route |
| Styling | Tailwind v4 + CSS variable design tokens (`app/globals.css`) | One palette, light/dark via `html[data-theme]`, no flash (inline init script) |
| Math | KaTeX, rendered server-side to HTML | No client math runtime for lesson text |
| Code | prism-react-renderer highlighting; copy, expected output, in-browser Run | |
| Python execution | Pyodide (WebAssembly CPython) loaded lazily from CDN | Executes untrusted learner code in the learner's own browser sandbox, never on our server |
| Simulators | React + SVG (diagrams) / Canvas (per-pixel, contours); no animation library | Exact control; numbers shown are numbers computed |
| Search | Static JSON index generated at build (`app/search-index.json/route.ts`), scored client-side | Searches modules, topics, every lesson section, formulas, code, glossary, simulators, papers |
| Progress | `lib/progress.ts`: localStorage store via `useSyncExternalStore`; optional push/pull to API | Works without accounts; union-merge never loses data |

### Content model

- `content/curriculum.ts` — 21 modules → topics (slug, level, summary, subtopics). Single source of truth for
  navigation, learning order (prev/next), roadmap and search.
- `content/lessons/<module>-<slug>.ts` — full lessons, registered in `content/lessons/index.ts`. Topics without
  a lesson render their syllabus with an explicit “being written” notice (no placeholder text pretending to be content).
- Blocks (`lib/types.ts`): p, h, math, list, code, callout, table, sim, steps, details, tabs, qa, quiz, exercise.
- Authoring contract: `docs/CONTENT_GUIDE.md`.

### Simulator framework

`components/sims/SimShell.tsx` provides `useStepper` (play / pause / step / back / reset / speed),
`Controls`, `Slider`, `Narration` and colour helpers. Each simulator is lazy-loaded via `SimEmbed.tsx`
and described in `registry.ts` (used by the Playground). Colour semantics: accent = positive / forward,
clay = negative / gradient / backward, gold = current focus.

## Backend (backend/)

FastAPI + SQLAlchemy 2. Endpoints: `GET/PUT /api/progress/{learnerId}`, `GET /api/stats`, `GET /health`.
SQLite by default, PostgreSQL through `DATABASE_URL`. Server-side union-merge mirrors the client's.
Input validation (id format, payload size, Pydantic schema). Tests in `backend/tests`.

Future backend responsibilities (see ROADMAP): accounts, isolated execution service for heavy examples
(container sandbox with CPU/memory/time limits — never in the API process), and the three project backends.
