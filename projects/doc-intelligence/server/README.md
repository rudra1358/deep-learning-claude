# Document Intelligence: API server (Express + MongoDB)

The API the browser calls. It owns the data (MongoDB) and the background jobs, and calls the Python ML
service (`../ml-service`, port 8200) for everything that needs a model.

## Run it

```bash
# MongoDB must be running (default mongodb://127.0.0.1:27017) and the ML service started:
#   python ../scripts/dev_api.py
npm install
npm start              # http://localhost:4200/api/health     (npm run dev = restart on file changes)
npm test               # real MongoDB (database "docintel-test", wiped) + a fake ML service
```

Settings (environment variables, or a `.env` file — copy `.env.example`): `PORT` (4200), `MONGO_URL`,
`ML_URL` (http://127.0.0.1:8200), `DATA_DIR` (../data — must be the same folder the ML service uses),
`MAX_UPLOAD_MB` (50), `CLIENT_DIST` (../client/dist: if it exists, the server also serves the built UI).

## The big picture (say this first in an interview)

1. **Three tiers.** React (client) → Express (this) → Python (ml-service). The browser never talks to Python.
2. **This server is the system of record.** It decides what exists (MongoDB) and when work happens (job
   queues). Python only does maths on what it is given — if it is down, nothing is lost; search and
   answering just return 503 until it is back.
3. **Same API shape as before.** Errors are always `{"detail": "..."}` with a meaningful status (404, 409,
   413, 415, 422, 503). Errors from Python are passed through unchanged.

```
POST   /api/documents            multer → data/uploads, Mongo doc "queued", 202 → jobs.js
GET    /api/documents            list (newest first)            GET /api/documents/:id   one
GET    /api/documents/:id/pages  page texts (viewer)            GET /api/documents/:id/file  download
DELETE /api/documents/:id        409 while processing; ML index first, then file, pages, doc
POST   /api/search               validate → ML /search
POST   /api/ask                  chat session (Mongo) + history → ML /ask → store both messages
GET    /api/chats  · GET /api/chats/:id · DELETE /api/chats/:id
GET    /api/health               {mongo, ml, ...ML health}        GET /api/metrics   ML /metrics
```

## File by file

| file | what it does |
|---|---|
| `package.json` | dependencies (express 5, mongoose, multer, cors, dotenv, undici) and scripts |
| `src/index.js` | entry point: connect MongoDB → `recover()` interrupted jobs → listen |
| `src/app.js` | `createApp()`: middleware, routes, static UI + SPA fallback, the one error handler |
| `src/config.js` | every setting with its default; paths resolved from this folder |
| `src/db.js` | Mongoose connection; JSON shape `id` instead of `_id`; ObjectId check |
| `src/models/Document.js` | status (`queued → processing → ready/failed`), enrich status, counts, label, entities, summary |
| `src/models/Page.js` | one page's text; separate collection so listing documents stays fast |
| `src/models/ChatSession.js` | a conversation with its messages embedded (sources + meta per answer) |
| `src/ml.js` | the only file that calls the ML service; error forwarding; long timeouts for slow model work |
| `src/jobs.js` | two in-memory queues (index, enrich), one job at a time each; crash recovery |
| `src/routes/documents.js` | upload / list / get / pages / file / delete |
| `src/routes/qa.js` | search, ask (+ chat history), chats |
| `src/routes/system.js` | health, metrics |
| `tests/api.test.js` | the whole HTTP flow with a fake ML service (a small Express app inside the test) |
| `Dockerfile` | builds the React client, then runs this server serving API + UI on port 4200 |

## Design notes worth knowing

* **Why two queues?** A slow summary (enrich) must never keep a new upload from becoming searchable (index).
* **Why `undici` instead of plain `fetch`?** Node's fetch stops waiting for response headers after 5 minutes;
  summarising a long PDF or a CPU LLM answer can take longer. A custom `Agent` removes that limit; each call
  sets its own timeout.
* **Why is `path` hidden in JSON?** It is a server file path — not the browser's business.
* **Scaling up:** replace the in-memory queues with BullMQ + Redis (jobs survive restarts and can run on many
  workers); the `status` field in MongoDB stays the contract between the API and the workers.
