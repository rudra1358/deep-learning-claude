// History, per-run details, CSV export and dashboard analytics - all straight from MongoDB.
// (No ML service involved: once a run is stored, Python is not needed to look at it.)
import fs from "node:fs";
import path from "node:path";
import { Router } from "express";
import { config } from "../config.js";
import { HttpError } from "../errors.js";
import { HEAVY_FIELDS, Run } from "../models/Run.js";

export const runsRouter = Router();

const runDir = (id) => path.join(config.dataDir, "runs", String(id));
const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

async function runOr404(id, projection) {
  const run = await Run.findById(id, projection);
  if (!run) throw new HttpError(404, "run not found");
  return run;
}

function intParam(value, fallback, min, max, name) {
  if (value === undefined || value === "") return fallback;
  const n = Number(value);
  if (!Number.isInteger(n) || n < min || n > max) throw new HttpError(422, `${name} must be an integer in [${min}, ${max}]`);
  return n;
}

// GET /api/runs?kind=&label=&q=&limit=&offset=  - search history
runsRouter.get("/runs", async (req, res) => {
  const { kind, label, q } = req.query;
  const limit = intParam(req.query.limit, 20, 1, 100, "limit");
  const offset = intParam(req.query.offset, 0, 0, Number.MAX_SAFE_INTEGER, "offset");

  const filter = {};
  if (kind) filter.kind = String(kind);
  if (q) filter.source_name = { $regex: escapeRegex(String(q)), $options: "i" }; // "contains", ignore case
  if (label) filter.$or = [{ "detections.label": String(label) }, { "tracks.label": String(label) }];

  const [total, items] = await Promise.all([
    Run.countDocuments(filter),
    Run.find(filter, HEAVY_FIELDS).sort({ created_at: -1, _id: -1 }).skip(offset).limit(limit),
  ]);
  res.json({ total, items });
});

runsRouter.get("/runs/:id", async (req, res) => {
  res.json(await runOr404(req.params.id, HEAVY_FIELDS));
});

runsRouter.delete("/runs/:id", async (req, res) => {
  const run = await runOr404(req.params.id, "status media_path");
  if (run.status === "queued" || run.status === "running") throw new HttpError(409, "run is still processing");
  if (run.media_path) fs.rmSync(run.media_path, { force: true });
  fs.rmSync(runDir(run._id), { recursive: true, force: true });
  await run.deleteOne();
  res.status(204).end();
});

// Boxes of an image run
runsRouter.get("/runs/:id/detections", async (req, res) => {
  const run = await runOr404(req.params.id, "detections");
  res.json(run.detections.map((d) => ({ label: d.label, score: d.score, box: d.box })));
});

// Per-frame tracks of a video run (written to disk by the ML service)
runsRouter.get("/runs/:id/frames", async (req, res) => {
  const run = await runOr404(req.params.id, "status");
  const file = path.join(runDir(run._id), "frames.json");
  if (run.status !== "done" || !fs.existsSync(file)) throw new HttpError(409, `run status is '${run.status}'`);
  res.sendFile(file);
});

// The video: the browser-playable WebM made by the ML service if present, else the original upload.
// res.sendFile supports HTTP Range requests, so seeking in the <video> element works.
runsRouter.get("/runs/:id/video", async (req, res) => {
  const run = await runOr404(req.params.id, "status media_path");
  const playback = path.join(runDir(run._id), "playback.webm");
  if (run.status === "done" && fs.existsSync(playback)) return res.sendFile(playback, { headers: { "Content-Type": "video/webm" } });
  if (!run.media_path || !fs.existsSync(run.media_path)) throw new HttpError(404, "no stored video for this run");
  res.sendFile(run.media_path);
});

// Line-crossing events, in time order
runsRouter.get("/runs/:id/events", async (req, res) => {
  const run = await runOr404(req.params.id, "events");
  res.json([...run.events].sort((a, b) => a.t_seconds - b.t_seconds).map((e) => e.toObject()));
});

// Per-second peak count of each class - drives the "counts over time" chart.
// Turns rows [{t_seconds, label, count}] into {labels, t: [seconds], series: {label: [count per second]}}.
runsRouter.get("/runs/:id/timeseries", async (req, res) => {
  const run = await runOr404(req.params.id, "samples");
  const t = [...new Set(run.samples.map((s) => s.t_seconds))].sort((a, b) => a - b);
  const labels = [...new Set(run.samples.map((s) => s.label))].sort();
  const index = new Map(t.map((sec, i) => [sec, i]));
  const series = Object.fromEntries(labels.map((l) => [l, new Array(t.length).fill(0)]));
  for (const s of run.samples) series[s.label][index.get(s.t_seconds)] = s.count;
  res.json({ labels, t, series });
});

runsRouter.get("/runs/:id/export.csv", async (req, res) => {
  const run = await runOr404(req.params.id);
  const rows = [];
  if (run.kind === "image") {
    rows.push(["label", "score", "x1", "y1", "x2", "y2"]);
    for (const d of run.detections) rows.push([d.label, d.score, ...d.box]);
  } else {
    rows.push(["track_id", "label", "first_frame", "last_frame", "seconds_visible", "max_score"]);
    const fps = run.fps || 30;
    for (const tr of [...run.tracks].sort((a, b) => a.track_id - b.track_id)) {
      const seconds = Math.round(((tr.last_frame - tr.first_frame + 1) / fps) * 100) / 100;
      rows.push([tr.track_id, tr.label, tr.first_frame, tr.last_frame, seconds, tr.max_score]);
    }
  }
  res.set("Content-Disposition", `attachment; filename="run-${run.id}.csv"`);
  res.type("text/csv").send(rows.map((r) => r.map(csvCell).join(",")).join("\r\n") + "\r\n");
});

function csvCell(v) {
  const s = String(v);
  return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

// GET /api/analytics/overview - dashboard totals across every run.
// The counting happens inside MongoDB with aggregation pipelines ($unwind = one row per array
// element, $group = GROUP BY), not by loading every run into Node.
runsRouter.get("/analytics/overview", async (_req, res) => {
  const [kinds, detLabels, trkLabels, crossings, latency] = await Promise.all([
    Run.aggregate([{ $group: { _id: "$kind", n: { $sum: 1 } } }]),
    Run.aggregate([{ $unwind: "$detections" }, { $group: { _id: "$detections.label", n: { $sum: 1 } } }]),
    Run.aggregate([{ $unwind: "$tracks" }, { $group: { _id: "$tracks.label", n: { $sum: 1 } } }]),
    Run.aggregate([
      { $unwind: "$events" },
      { $group: { _id: { label: "$events.label", direction: "$events.direction" }, n: { $sum: 1 } } },
    ]),
    Run.aggregate([
      { $match: { status: "done", avg_latency_ms: { $gt: 0 } } },
      { $group: { _id: null, avg: { $avg: "$avg_latency_ms" } } },
    ]),
  ]);

  const objects = {};
  for (const { _id, n } of [...detLabels, ...trkLabels]) objects[_id] = (objects[_id] ?? 0) + n;
  const crossingsOut = { in: {}, out: {} };
  for (const { _id, n } of crossings) crossingsOut[_id.direction][_id.label] = n;
  const avg = latency[0]?.avg;

  res.json({
    runs_by_kind: Object.fromEntries(kinds.map((k) => [k._id, k.n])),
    objects_by_label: Object.fromEntries(Object.entries(objects).sort((a, b) => b[1] - a[1])),
    crossings: crossingsOut,
    avg_latency_ms: avg ? Math.round(avg * 100) / 100 : null,
  });
});
