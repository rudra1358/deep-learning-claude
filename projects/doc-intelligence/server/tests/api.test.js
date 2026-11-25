// API tests: the REAL server + a REAL MongoDB (database "docintel-test", wiped first) + a FAKE ML service.
//
// The fake ML service is a tiny Express app that behaves like ml-service/ without any models:
// it "indexes" a .txt file by reading it, "searches" by word overlap, "answers" with the first
// sentence containing a question word. That is enough to test the plumbing the server owns:
// uploads, background jobs, MongoDB records, chat history, error forwarding and crash recovery.
// (Model quality is measured by training/; the ML endpoints have their own pytest suite.)
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, before, test } from "node:test";
import express from "express";
import mongoose from "mongoose";
import request from "supertest";

// ---------------------------------------------------------------- the fake ML service
const index = new Map(); // document_id -> {filename, text}
const removed = [];
let enrichGate = null; // a Promise: while set, /enrich waits for it (simulates a slow summariser)
let mlBroken = false; // simulate the ML service answering 503

const words = (t) => t.toLowerCase().match(/[a-z0-9]+/g) ?? [];

const fake = express();
fake.use(express.json());
fake.use((_req, res, next) => (mlBroken ? res.status(503).json({ detail: "generator disabled" }) : next()));
fake.get("/health", (_req, res) => res.json({ status: "ok", chunks_indexed: index.size, models_loaded: ["embedder"] }));
fake.get("/metrics", (_req, res) => res.json({ qa: { tuned: { no_answer_margin: 1.5 } } }));
fake.post("/index/documents", (req, res) => {
  const text = fs.readFileSync(req.body.path, "utf8");
  if (text.includes("SCANNED")) return res.status(422).json({ detail: "no text found (is this a scanned PDF?)" });
  index.set(req.body.document_id, { filename: req.body.filename, text });
  res.json({ document_id: req.body.document_id, pages: [{ number: 1, text }], num_chunks: 1 });
});
fake.delete("/index/documents/:id", (req, res) => {
  removed.push(req.params.id);
  res.json({ removed_chunks: index.delete(req.params.id) ? 1 : 0 });
});
fake.post("/enrich", async (req, res) => {
  if (enrichGate) await enrichGate;
  res.json({ label: null, label_score: null, entities: null, summary: req.body.text.split(".")[0] + "." });
});
function hits(query, ids) {
  const q = new Set(words(query));
  return [...index]
    .filter(([id]) => !ids || ids.includes(id))
    .map(([id, d]) => ({ id, d, score: words(d.text).filter((w) => q.has(w)).length }))
    .filter((h) => h.score > 0)
    .sort((a, b) => b.score - a.score)
    .map((h, i) => ({ n: i + 1, chunk_id: i + 1, document_id: h.id, filename: h.d.filename, page_start: 1,
      page_end: 1, char_start: 0, score: h.score, ranks: {}, text: h.d.text }));
}
fake.post("/search", (req, res) => res.json({ hits: hits(req.body.query, req.body.document_ids), timing: {} }));
fake.post("/ask", (req, res) => {
  const sources = hits(req.body.question, req.body.document_ids);
  const answer = sources.length ? sources[0].text.split(".")[0] + "." : null;
  res.json({ answer, found: Boolean(answer), mode: req.body.mode, cited: answer ? [1] : [], sources,
    timing: { history_turns: req.body.history.length } }); // echo how much history the server sent
});

// ---------------------------------------------------------------- setup
let app, mlServer, tmp, recover, Document;

before(async () => {
  mlServer = fake.listen(0);
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), "docintel-"));
  // config.js reads the environment once, when it is first imported -> set it BEFORE importing the app
  Object.assign(process.env, {
    ML_URL: `http://127.0.0.1:${mlServer.address().port}`,
    MONGO_URL: "mongodb://127.0.0.1:27017/docintel-test",
    DATA_DIR: tmp,
    MAX_UPLOAD_MB: "1",
    CLIENT_DIST: path.join(tmp, "no-client"),
  });
  const { createApp } = await import("../src/app.js");
  const { connectDb } = await import("../src/db.js");
  ({ recover } = await import("../src/jobs.js"));
  ({ Document } = await import("../src/models/Document.js"));
  await connectDb(process.env.MONGO_URL);
  await mongoose.connection.dropDatabase();
  app = createApp();
});

after(async () => {
  await mongoose.connection.dropDatabase();
  await mongoose.disconnect();
  mlServer.close();
  fs.rmSync(tmp, { recursive: true, force: true });
});

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitFor(id, done, timeout = 5000) {
  const t0 = Date.now();
  for (;;) {
    const d = (await request(app).get(`/api/documents/${id}`)).body;
    if (done(d)) return d;
    if (Date.now() - t0 > timeout) throw new Error(`timeout: ${JSON.stringify(d)}`);
    await sleep(20);
  }
}
const finished = (d) => ["ready", "failed"].includes(d.status) && ["done", "failed", "skipped"].includes(d.enrich_status);

function uploadTexts(files) {
  let r = request(app).post("/api/documents");
  for (const [name, text] of files) r = r.attach("files", Buffer.from(text), name);
  return r;
}

const SPACE = "Neil Armstrong was the first person to walk on the Moon. The Saturn rocket launched Apollo.";
const BREAD = "Bake the bread in a hot oven for forty minutes. Yeast makes dough rise.";

// ---------------------------------------------------------------- tests
test("upload -> index -> enrich -> search -> ask with chat history -> delete", async () => {
  const up = await uploadTexts([["space.txt", SPACE], ["bread.md", BREAD]]);
  assert.equal(up.status, 202, JSON.stringify(up.body));
  assert.equal(up.body.length, 2);
  assert.equal(up.body[0].status, "queued");
  assert.equal(up.body[0].path, undefined); // server file paths are not exposed
  const [space, bread] = await Promise.all(up.body.map((d) => waitFor(d.id, finished)));
  assert.equal(space.status, "ready");
  assert.equal(space.num_pages, 1);
  assert.equal(space.summary, "Neil Armstrong was the first person to walk on the Moon.");
  assert.match(space.id, /^[0-9a-f]{24}$/);
  assert.ok(space.created_at);

  const list = await request(app).get("/api/documents");
  assert.deepEqual(new Set(list.body.map((d) => d.id)), new Set([space.id, bread.id]));
  const pages = await request(app).get(`/api/documents/${space.id}/pages`);
  assert.deepEqual(pages.body, [{ number: 1, text: SPACE }]);
  const file = await request(app).get(`/api/documents/${space.id}/file`);
  assert.equal(file.status, 200);
  assert.match(file.headers["content-disposition"], /space\.txt/);

  const s = await request(app).post("/api/search").send({ query: "Moon rocket", document_ids: [bread.id] });
  assert.equal(s.status, 200);
  assert.deepEqual(s.body.hits, []); // the document filter reached the ML service
  const s2 = await request(app).post("/api/search").send({ query: "Moon rocket" });
  assert.equal(s2.body.hits[0].document_id, space.id);

  const a1 = await request(app).post("/api/ask").send({ question: "Who walked on the Moon first?" });
  assert.equal(a1.status, 200);
  assert.match(a1.body.answer, /Armstrong/);
  assert.equal(a1.body.timing.history_turns, 0);
  const a2 = await request(app).post("/api/ask").send({ question: "How long does bread bake in the oven?", session_id: a1.body.session_id });
  assert.equal(a2.body.session_id, a1.body.session_id);
  assert.equal(a2.body.timing.history_turns, 2); // the previous question + answer were sent along
  const chat = await request(app).get(`/api/chats/${a1.body.session_id}`);
  assert.deepEqual(chat.body.messages.map((m) => m.role), ["user", "assistant", "user", "assistant"]);
  assert.ok(chat.body.messages[1].sources.length);
  assert.equal(chat.body.messages[1].meta.found, true);
  assert.equal(chat.body.title, "Who walked on the Moon first?");
  const chats = await request(app).get("/api/chats");
  assert.equal(chats.body[0].id, a1.body.session_id);
  assert.equal(chats.body[0].messages, undefined); // the list stays light

  const none = await request(app).post("/api/ask").send({ question: "zebra quantum giraffe?" });
  assert.equal(none.body.found, false);
  const noneChat = await request(app).get(`/api/chats/${none.body.session_id}`);
  assert.equal(noneChat.body.messages[1].content, "No answer found in the documents.");
  assert.equal((await request(app).delete(`/api/chats/${none.body.session_id}`)).status, 204);
  assert.equal((await request(app).get(`/api/chats/${none.body.session_id}`)).status, 404);

  const del = await request(app).delete(`/api/documents/${space.id}`);
  assert.equal(del.status, 204);
  assert.ok(removed.includes(space.id)); // removed from the search index too
  assert.equal((await request(app).get(`/api/documents/${space.id}`)).status, 404);
  assert.equal(fs.readdirSync(path.join(tmp, "uploads")).length, 1); // the file is gone, bread.md stays
});

test("failed indexing is recorded; enrichment is skipped, not left pending", async () => {
  const up = await uploadTexts([["scan.txt", "SCANNED image only"]]);
  const d = await waitFor(up.body[0].id, finished);
  assert.equal(d.status, "failed");
  assert.equal(d.enrich_status, "skipped");
  assert.match(d.error, /no text/);
});

test("a slow enrichment does not block indexing of newer uploads", async () => {
  let release;
  enrichGate = new Promise((r) => (release = r));
  try {
    const first = (await uploadTexts([["a.txt", "Owls hunt mice at night."]])).body[0];
    await waitFor(first.id, (d) => d.enrich_status === "running"); // the enrich queue is now stuck
    const second = (await uploadTexts([["b.txt", "Kittens enjoy chasing red laser dots."]])).body[0];
    const d = await waitFor(second.id, (x) => x.status === "ready");
    assert.equal(d.enrich_status, "pending"); // searchable even though enrichment is busy
    // a document being enriched cannot be deleted yet
    assert.equal((await request(app).delete(`/api/documents/${first.id}`)).status, 409);
  } finally {
    enrichGate = null;
    release();
  }
});

test("interrupted jobs are re-queued on startup", async () => {
  const p1 = path.join(tmp, "uploads", "stuck.txt");
  const p2 = path.join(tmp, "uploads", "waiting.txt");
  fs.writeFileSync(p1, "The lighthouse keeper lit the lamp every evening.");
  fs.writeFileSync(p2, "Bees make honey in summer.");
  // as if the server crashed: one document half-indexed, one waiting in the (lost) enrich queue
  const stuck = await Document.create({ filename: "stuck.txt", path: p1, status: "processing" });
  const waiting = await Document.create({ filename: "waiting.txt", path: p2, status: "ready", enrich_status: "pending" });
  assert.ok((await recover()) >= 2);
  const d1 = await waitFor(stuck.id, finished);
  assert.equal(d1.status, "ready");
  assert.equal(d1.enrich_status, "done");
  assert.equal((await waitFor(waiting.id, finished)).enrich_status, "done");
});

test("validation and error forwarding", async () => {
  const bad = await uploadTexts([["virus.exe", "MZ"]]);
  assert.equal(bad.status, 415);
  assert.match(bad.body.detail, /unsupported type/);
  const big = await uploadTexts([["big.txt", "x".repeat(2 * 2 ** 20)]]); // limit is 1 MB in tests
  assert.equal(big.status, 413);
  assert.equal((await request(app).post("/api/documents")).status, 422);

  assert.equal((await request(app).get("/api/documents/123")).status, 404); // not an ObjectId
  assert.equal((await request(app).get(`/api/documents/${"a".repeat(24)}`)).status, 404);
  assert.equal((await request(app).post("/api/search").send({ query: "" })).status, 422);
  assert.equal((await request(app).post("/api/search").send({ query: "x", mode: "magic" })).status, 422);
  assert.equal((await request(app).post("/api/ask").send({ question: "hi", session_id: "a".repeat(24) })).status, 404);
  assert.equal((await request(app).get("/api/nope")).status, 404);

  mlBroken = true; // errors from Python reach the browser with the same status and message
  try {
    const r = await request(app).post("/api/ask").send({ question: "anything?", mode: "generative" });
    assert.equal(r.status, 503);
    assert.equal(r.body.detail, "generator disabled");
  } finally {
    mlBroken = false;
  }
});

test("health reports MongoDB and the ML service separately", async () => {
  const ok = await request(app).get("/api/health");
  assert.equal(ok.body.status, "ok");
  assert.equal(ok.body.mongo, "ok");
  assert.equal(ok.body.ml, "ok");
  assert.deepEqual((await request(app).get("/api/metrics")).body, { qa: { tuned: { no_answer_margin: 1.5 } } });

  const { config } = await import("../src/config.js");
  const real = config.mlUrl;
  config.mlUrl = "http://127.0.0.1:9"; // nothing listens there
  try {
    const down = await request(app).get("/api/health");
    assert.equal(down.body.status, "degraded");
    assert.equal(down.body.ml, "down");
    assert.equal((await request(app).post("/api/search").send({ query: "x" })).status, 503);
  } finally {
    config.mlUrl = real;
  }
});
