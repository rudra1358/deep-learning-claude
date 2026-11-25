// Image detection, model comparison and video upload.
//
// The pattern for every endpoint: validate the request here -> ask the Python ML service to do
// the deep-learning work -> store the result in MongoDB -> answer the browser.
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { Router } from "express";
import multer from "multer";
import { config } from "../config.js";
import { HttpError } from "../errors.js";
import { startVideoJob } from "../jobs.js";
import { fileForm, mlRequest } from "../ml.js";
import { Run } from "../models/Run.js";

export const detectRouter = Router();

const VIDEO_EXT = new Set([".mp4", ".avi", ".mov", ".mkv", ".webm"]);
const limits = { fileSize: config.maxUploadMb * 2 ** 20 };

// Images are small: keep them in memory and forward them to the ML service.
const imageUpload = multer({ storage: multer.memoryStorage(), limits });

// Videos can be hundreds of MB: stream them straight to disk (never the whole file in RAM).
// The ML service reads the file from the same data/uploads folder.
const videoUpload = multer({
  limits,
  storage: multer.diskStorage({
    destination(_req, _file, cb) {
      const dir = path.join(config.dataDir, "uploads");
      fs.mkdirSync(dir, { recursive: true });
      cb(null, dir);
    },
    filename(_req, file, cb) {
      cb(null, crypto.randomUUID().replaceAll("-", "") + path.extname(file.originalname).toLowerCase());
    },
  }),
  fileFilter(_req, file, cb) {
    const ext = path.extname(file.originalname).toLowerCase();
    if (!VIDEO_EXT.has(ext)) return cb(new HttpError(415, `unsupported video type '${ext}'`));
    cb(null, true);
  },
});

function requireFile(req) {
  if (!req.file) throw new HttpError(422, "file is required");
}

// "0.3" -> 0.3, with a default when the form field is missing
function num(value, fallback) {
  if (value === undefined || value === "") return fallback;
  const n = Number(value);
  if (!Number.isFinite(n)) throw new HttpError(422, `not a number: ${value}`);
  return n;
}

// POST /api/detect/image  (multipart: file, model?, conf, iou, save)
detectRouter.post("/detect/image", imageUpload.single("file"), async (req, res) => {
  requireFile(req);
  const conf = num(req.body.conf, 0.25);
  const iou = num(req.body.iou, 0.45);
  const result = await mlRequest("/detect/image", {
    method: "POST",
    form: fileForm(req.file, { model: req.body.model, conf, iou }),
  });

  let runId = null;
  if (req.body.save !== "false") {
    const run = await Run.create({
      kind: "image",
      model: result.model,
      source_name: req.file.originalname || "image",
      width: result.width,
      height: result.height,
      frames: 1,
      avg_latency_ms: result.timing.total_ms,
      conf,
      summary: { counts: result.counts, timing: result.timing },
      detections: result.detections.map((d) => ({ label: d.label, score: d.score, box: d.box })),
    });
    runId = run.id;
  }
  res.json({ run_id: runId, ...result });
});

// POST /api/compare  (multipart: file, models="a,b", conf, iou) - nothing is stored
detectRouter.post("/compare", imageUpload.single("file"), async (req, res) => {
  requireFile(req);
  const result = await mlRequest("/compare", {
    method: "POST",
    form: fileForm(req.file, {
      models: req.body.models,
      conf: num(req.body.conf, 0.25),
      iou: num(req.body.iou, 0.45),
    }),
  });
  res.json(result);
});

// POST /api/detect/video  (multipart: file, model?, conf, line?) -> 202 + the queued run
detectRouter.post("/detect/video", videoUpload.single("file"), async (req, res) => {
  requireFile(req);
  let run = null;
  try {
    const conf = num(req.body.conf, 0.25);
    if (!(conf > 0 && conf < 1)) throw new HttpError(422, "conf must be in (0, 1)");
    const line = parseLine(req.body.line);
    const model = req.body.model || (await mlRequest("/health", { timeoutMs: 5000 })).default_model;
    run = await Run.create({
      kind: "video",
      model,
      source_name: req.file.originalname || req.file.filename,
      status: "queued",
      progress: 0,
      conf,
      line,
      media_path: req.file.path,
    });
    await startVideoJob(run); // the ML service checks the model name etc. - errors land in catch
  } catch (err) {
    // Nothing was queued: don't leave a half-created run or an orphaned upload behind.
    if (run) await Run.deleteOne({ _id: run._id });
    fs.rmSync(req.file.path, { force: true });
    throw err;
  }
  res.status(202).json(run);
});

// line = JSON string "[x1, y1, x2, y2]", each in 0..1 (or missing = no counting line)
function parseLine(raw) {
  if (!raw) return null;
  try {
    const v = JSON.parse(raw).map(Number);
    if (v.length === 4 && v.every((x) => x >= 0 && x <= 1)) return v;
  } catch {
    /* fall through */
  }
  throw new HttpError(422, "line must be a JSON list of 4 numbers in [0, 1]");
}
