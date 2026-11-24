# Deep Learning University

An interactive deep learning curriculum: from vectors and derivatives to backpropagation, CNNs, LSTMs and
Transformers. Every concept is taught in a fixed 16-part format (intuition → mechanism → derivation → simulator →
NumPy from scratch → PyTorch/Keras → practice), and every simulator runs the real computation.

## Repository layout

```
web/        Next.js 15 + React 19 + TypeScript + Tailwind v4 — the website
  app/            routes (home, curriculum, roadmap, learn/[module]/[topic], playground, visualizer,
                  math-lab, architectures, code, practice, projects, papers, glossary, resources, progress)
  content/        curriculum.ts (21 modules), lessons/*.ts (full lessons), glossary, papers, projects
  components/     content renderer, lesson layout, simulators (components/sims), page components
  lib/            types (lesson schema), progress store, search, Pyodide runner, sim math (lib/sim/*)
backend/    FastAPI progress-sync API (SQLite by default, PostgreSQL via DATABASE_URL)
docs/       ARCHITECTURE.md, CONTENT_GUIDE.md, ROADMAP.md, STATUS.md
```

## Running locally

Requirements: Node 20+ (tested on 24), Python 3.11+ (tested on 3.14).

```bash
cd web
npm install
npm run dev          # http://localhost:3000
```

Production build: `npm run build && npm start`.

### Optional: progress-sync backend

Progress is stored in the browser by default. To also sync it to a server:

```bash
python -m venv .venv
.venv/Scripts/pip install -r backend/requirements.txt      # (bin/ instead of Scripts/ on macOS/Linux)
cd backend && ../.venv/Scripts/uvicorn app.main:app --port 8000
```

Then start the web app with `NEXT_PUBLIC_API_URL=http://localhost:8000`. Tests: `cd backend && pytest`.
Set `DATABASE_URL=postgresql+psycopg://…` (and install `psycopg[binary]`) to use PostgreSQL.

### Verifying lesson code

All lesson Python was executed during authoring with NumPy 2.5, PyTorch 2.15 (CPU), torchvision 0.29 and
scikit-learn 1.9 (plus `tokenizers` for the tokenization lesson); re-verify everything with
`cd web && npm run check:lessons`, which re-runs every example and diffs its output.
outputs shown in lessons are the real outputs. TensorFlow/Keras examples are marked “not executed in CI”
because TensorFlow was not installable in the authoring environment.

## Status

See [docs/STATUS.md](docs/STATUS.md) for what is complete and [docs/ROADMAP.md](docs/ROADMAP.md) for what is next.
