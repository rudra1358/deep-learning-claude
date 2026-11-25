// Server tests: real MongoDB (a throw-away "vision-test" database) + a FAKE ML service.
//
// The fake ML service is a tiny Express app that answers like the Python one, so these tests
// check the server's own job - validation, storage, history, analytics, video-job polling and
// the WebSocket relay - without loading any model. (Python is tested in ml-service/tests.)
//
// Needs MongoDB on localhost (or set TEST_MONGO_URL).   Run:  npm test
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { after, before, test } from "node:test";
import express from "express";
import mongoose from "mongoose";
import request from "supertest";
import { WebSocket, WebSocketServer } from "ws";

// ------------------------------------------------------------------ fake ML service
const jobs = new Map();
const fakeResult = {
  model: "fake",
  provider: "CPUExecutionProvider",
  width: 1280,
  height: 640,
  detections: [
    { box: [256, 192, 384, 448], score: 0.92, cls: 0, label: "person" },
    { box: [800, 280, 1120, 440], score: 0.81, cls: 1, label: "car" },
  ],
  counts: { person: 1, car: 1 },
  timing: { preprocess_ms: 1, inference_ms: 2, postprocess_ms: 1, total_ms: 4 },
};
const sessionSummary = { frames: 2, unique_objects: { person: 1 }, avg_latency_ms: 3 };

const ml = express();
ml.use(express.json());
ml.get("/health", (_q, r) => r.json({ status: "ok", providers: ["CPUExecutionProvider"], models: ["fake"], default_model: "fake" }));
ml.get("/models", (_q, r) => r.json([{ name: "fake", loaded: true, metrics: null }]));
ml.get("/metrics", (_q, r) => r.json({ models: {} }));
ml.post("/detect/image", (req, res) => {
  // multipart body is not parsed here; the model name check is all we need
  let raw = "";
  req.on("data", (c) => (raw += c));
  req.on("end", () => {
    if (raw.includes('name="model"\r\n\r\nnope')) return res.status(404).json({ detail: "model 'nope' not found" });
    res.json(fakeResult);
  });
});
ml.post("/compare", (_q, r) => r.json({ results: [fakeResult, fakeResult] }));
ml.post("/jobs/video", (req, res) => {
  if (req.body.model === "nope") return res.status(404).json({ detail: "model 'nope' not found" });
  jobs.set(req.body.run_id, { polls: 0, broken: req.body.video_path.endsWith(".avi") });
  res.status(202).json({ run_id: req.body.run_id, status: "queued", progress: 0, model: req.body.model });
});
ml.get("/jobs/:id", (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ detail: "job not found" });
  job.polls += 1;
  const base = { run_id: req.params.id, model: "fake", width: 320, height: 160, fps: 10 };
  if (job.polls < 2) return res.json({ ...base, status: "running", progress: 0.5 });
  if (job.broken) return res.json({ ...base, status: "failed", progress: 0.5, error: "could not open video" });
  // like the Python worker: files on disk + result in the response
  const dir = path.join(dataDir, "runs", req.params.id);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "frames.json"), JSON.stringify({ fps: 10, width: 320, height: 160, frames: [{ t: 0, tracks: [] }] }));
  fs.writeFileSync(path.join(dir, "playback.webm"), "webm-bytes");
  res.json({
    ...base,
    status: "done",
    progress: 1,
    result: {
      frames: 20,
      avg_latency_ms: 5,
      summary: { unique_objects: { person: 1, car: 1 } },
      tracks: [
        { track_id: 2, label: "car", first_frame: 1, last_frame: 20, max_score: 0.8 },
        { track_id: 1, label: "person", first_frame: 1, last_frame: 10, max_score: 0.9 },
      ],
      events: [
        { track_id: 1, label: "person", direction: "in", t_seconds: 1.5 },
        { track_id: 2, label: "car", direction: "out", t_seconds: 0.5 },
      ],
      samples: [
        { t_seconds: 0, label: "person", count: 1 },
        { t_seconds: 1, label: "car", count: 2 },
      ],
    },
  });
});
ml.delete("/jobs/:id", (req, res) => {
  jobs.delete(req.params.id);
  res.status(204).end();
});

let mlServer, appServer, baseUrl, app, dataDir;

before(async () => {
  const mlHttp = http.createServer(ml);
  const mlWs = new WebSocketServer({ server: mlHttp, path: "/stream" });
  mlWs.on("connection", (ws) => {
    let frames = 0;
    ws.on("message", (data, isBinary) => {
      if (isBinary) {
        frames += 1;
        return ws.send(JSON.stringify({ type: "result", frame: frames, tracks: [], occupancy: { person: 1 }, events: [] }));
      }
      const msg = JSON.parse(data.toString());
      if (msg.type === "config") ws.send(JSON.stringify({ type: "ready", model: "fake", provider: "CPU" }));
      if (msg.type === "stop") {
        const run = frames
          ? { kind: "stream", model: "fake", source_name: "webcam", frames, summary: sessionSummary,
              tracks: [{ track_id: 1, label: "person", first_frame: 1, last_frame: frames, max_score: 0.9 }],
              events: [], samples: [{ t_seconds: 0, label: "person", count: 1 }] }
          : null;
        ws.send(JSON.stringify({ type: "session", run }));
      }
    });
  });
  await new Promise((r) => mlHttp.listen(0, "127.0.0.1", r));
  mlServer = mlHttp;

  dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "va-server-"));
  Object.assign(process.env, {
    ML_URL: `http://127.0.0.1:${mlHttp.address().port}`,
    MONGO_URL: process.env.TEST_MONGO_URL ?? "mongodb://127.0.0.1:27017/vision-test",
    DATA_DIR: dataDir,
    JOB_POLL_MS: "20",
    CLIENT_DIST: path.join(dataDir, "no-client"),
  });
  // import AFTER setting env: config.js reads it at import time
  const { config } = await import("../src/config.js");
  const { connectDb } = await import("../src/db.js");
  const { createApp } = await import("../src/app.js");
  const { attachStreamRelay } = await import("../src/stream.js");
  await connectDb(config.mongoUrl);
  await mongoose.connection.dropDatabase();
  app = createApp();
  appServer = http.createServer(app);
  attachStreamRelay(appServer);
  await new Promise((r) => appServer.listen(0, "127.0.0.1", r));
  baseUrl = `127.0.0.1:${appServer.address().port}`;
});

after(async () => {
  await mongoose.connection.dropDatabase();
  await mongoose.disconnect();
  appServer.close();
  mlServer.close();
  fs.rmSync(dataDir, { recursive: true, force: true });
});

const jpeg = Buffer.from([0xff, 0xd8, 0xff, 0xd9]);

async function waitFinished(id) {
  for (let i = 0; i < 200; i++) {
    const run = (await request(app).get(`/api/runs/${id}`)).body;
    if (run.status === "done" || run.status === "failed") return run;
    await new Promise((r) => setTimeout(r, 20));
  }
  throw new Error("timeout");
}

// ------------------------------------------------------------------ tests
test("health combines MongoDB and ML service", async () => {
  const res = await request(app).get("/api/health").expect(200);
  assert.equal(res.body.status, "ok");
  assert.equal(res.body.mongo, "ok");
  assert.deepEqual(res.body.models, ["fake"]);
  await request(app).get("/api/models").expect(200);
  await request(app).get("/api/metrics").expect(200);
});

test("image detection is stored as a run with its detections", async () => {
  const res = await request(app)
    .post("/api/detect/image")
    .attach("file", jpeg, { filename: "street.jpg", contentType: "image/jpeg" })
    .field("conf", "0.3")
    .expect(200);
  assert.deepEqual(res.body.counts, { person: 1, car: 1 });
  assert.match(res.body.run_id, /^[0-9a-f]{24}$/);

  const run = (await request(app).get(`/api/runs/${res.body.run_id}`).expect(200)).body;
  assert.equal(run.kind, "image");
  assert.equal(run.conf, 0.3);
  assert.equal(run.detections, undefined); // big arrays are not part of the run JSON
  assert.match(run.created_at, /Z$/);
  const dets = (await request(app).get(`/api/runs/${run.id}/detections`).expect(200)).body;
  assert.equal(dets.length, 2);
  const csv = (await request(app).get(`/api/runs/${run.id}/export.csv`).expect(200)).text.trim().split(/\r?\n/);
  assert.equal(csv[0], "label,score,x1,y1,x2,y2");
  assert.equal(csv.length, 3);
});

test("save=false does not store anything; ML errors pass through", async () => {
  const res = await request(app).post("/api/detect/image").attach("file", jpeg, "a.jpg").field("save", "false").expect(200);
  assert.equal(res.body.run_id, null);
  const err = await request(app).post("/api/detect/image").attach("file", jpeg, "a.jpg").field("model", "nope").expect(404);
  assert.match(err.body.detail, /not found/);
  await request(app).post("/api/detect/image").expect(422); // no file
});

test("compare forwards to the ML service", async () => {
  const res = await request(app).post("/api/compare").attach("file", jpeg, "a.jpg").field("models", "fake,fake").expect(200);
  assert.equal(res.body.results.length, 2);
});

test("video upload -> job polled to completion -> results stored in MongoDB", async () => {
  const res = await request(app)
    .post("/api/detect/video")
    .attach("file", Buffer.from("fake mp4"), "clip.mp4")
    .field("line", JSON.stringify([0, 0.5, 1, 0.5]))
    .expect(202);
  assert.equal(res.body.status, "queued");
  const run = await waitFinished(res.body.id);
  assert.equal(run.status, "done", run.error);
  assert.equal(run.frames, 20);
  assert.deepEqual(run.line, [0, 0.5, 1, 0.5]);

  const events = (await request(app).get(`/api/runs/${run.id}/events`).expect(200)).body;
  assert.deepEqual(events.map((e) => e.t_seconds), [0.5, 1.5]); // sorted by time
  const ts = (await request(app).get(`/api/runs/${run.id}/timeseries`).expect(200)).body;
  assert.deepEqual(ts, { labels: ["car", "person"], t: [0, 1], series: { car: [0, 2], person: [1, 0] } });
  const csv = (await request(app).get(`/api/runs/${run.id}/export.csv`)).text.trim().split(/\r?\n/);
  assert.equal(csv[0], "track_id,label,first_frame,last_frame,seconds_visible,max_score");
  assert.equal(csv[1], "1,person,1,10,1,0.9"); // sorted by track id; 10 frames at 10 fps = 1 s
  await request(app).get(`/api/runs/${run.id}/frames`).expect(200);
  const video = await request(app).get(`/api/runs/${run.id}/video`).expect(200);
  assert.equal(video.headers["content-type"], "video/webm");
  assert.equal(jobs.has(run.id), false); // the server told the ML service to forget the job
});

test("failed video job is recorded on the run", async () => {
  const res = await request(app).post("/api/detect/video").attach("file", Buffer.from("x"), "broken.avi").expect(202);
  const run = await waitFinished(res.body.id);
  assert.equal(run.status, "failed");
  assert.match(run.error, /could not open/);
});

test("video validation: type, line, unknown model (nothing left behind)", async () => {
  await request(app).post("/api/detect/video").attach("file", Buffer.from("MZ"), "x.exe").expect(415);
  await request(app).post("/api/detect/video").attach("file", Buffer.from("00"), "x.mp4").field("line", "[0, 2, 1]").expect(422);
  const before = await mongoose.connection.collection("runs").countDocuments();
  await request(app).post("/api/detect/video").attach("file", Buffer.from("00"), "x.mp4").field("model", "nope").expect(404);
  assert.equal(await mongoose.connection.collection("runs").countDocuments(), before);
  assert.deepEqual(fs.readdirSync(path.join(dataDir, "uploads")).filter((f) => f.startsWith("x")), []);
});

test("live stream relay saves the session as a run", async () => {
  const ws = new WebSocket(`ws://${baseUrl}/api/stream`);
  const inbox = [];
  const next = () =>
    new Promise((resolve) => {
      if (inbox.length) return resolve(inbox.shift());
      ws.once("message", (d) => resolve(JSON.parse(d.toString())));
    });
  // sent before the relay's ML connection is open: must be buffered, not lost
  await new Promise((r) => ws.once("open", r));
  ws.send(JSON.stringify({ type: "config", model: "fake" }));
  assert.equal((await next()).type, "ready");
  for (let i = 0; i < 2; i++) {
    ws.send(jpeg, { binary: true });
    assert.equal((await next()).type, "result");
  }
  ws.send(JSON.stringify({ type: "stop" }));
  const saved = await next();
  assert.equal(saved.type, "saved");
  assert.deepEqual(saved.summary, sessionSummary);
  const run = (await request(app).get(`/api/runs/${saved.run_id}`).expect(200)).body;
  assert.equal(run.kind, "stream");
  assert.equal(run.frames, 2);
  ws.close();
});

test("history filters, analytics overview and delete", async () => {
  const byName = (await request(app).get("/api/runs").query({ q: "STREET" }).expect(200)).body;
  assert.ok(byName.total >= 1 && byName.items.every((r) => /street/i.test(r.source_name)));
  assert.ok((await request(app).get("/api/runs").query({ label: "car" })).body.total >= 2); // image + video
  assert.equal((await request(app).get("/api/runs").query({ label: "giraffe" })).body.total, 0);
  assert.equal((await request(app).get("/api/runs").query({ kind: "stream" })).body.total, 1);
  await request(app).get("/api/runs").query({ limit: 0 }).expect(422);
  const page = (await request(app).get("/api/runs").query({ limit: 2 })).body;
  assert.equal(page.items.length, 2);

  const ov = (await request(app).get("/api/analytics/overview").expect(200)).body;
  assert.equal(ov.runs_by_kind.image, 1);
  assert.ok(ov.objects_by_label.person >= 3); // image detection + video track + stream track
  assert.deepEqual(ov.crossings, { in: { person: 1 }, out: { car: 1 } });
  assert.ok(ov.avg_latency_ms > 0);

  const id = byName.items[0].id;
  await request(app).delete(`/api/runs/${id}`).expect(204);
  await request(app).get(`/api/runs/${id}`).expect(404);
  await request(app).get("/api/runs/not-an-id").expect(404);
  await request(app).get("/api/nothing-here").expect(404);
});
