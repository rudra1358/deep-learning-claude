// Upload, list, inspect and delete documents.  Mounted at /api/documents.
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { Router } from "express";
import multer from "multer";
import { config, SUPPORTED } from "../config.js";
import { isValidId } from "../db.js";
import { submit } from "../jobs.js";
import { HttpError, ml } from "../ml.js";
import { Document } from "../models/Document.js";
import { Page } from "../models/Page.js";

const uploadDir = path.join(config.dataDir, "uploads");
fs.mkdirSync(uploadDir, { recursive: true });

// multer streams each file straight to disk (never the whole file in RAM) under a random name,
// and stops with a 413 as soon as a file is bigger than the limit (see the error handler in app.js).
const upload = multer({
  storage: multer.diskStorage({
    destination: uploadDir,
    filename: (_req, file, cb) => cb(null, crypto.randomUUID() + path.extname(file.originalname).toLowerCase()),
  }),
  limits: { fileSize: config.maxUploadMb * 2 ** 20 },
  defParamCharset: "utf8", // keep non-English file names intact
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (SUPPORTED.includes(ext)) cb(null, true);
    else cb(new HttpError(415, `${file.originalname}: unsupported type; use ${JSON.stringify(SUPPORTED)}`));
  },
});

const router = Router();

async function findDoc(id) {
  const doc = isValidId(id) ? await Document.findById(id) : null;
  if (!doc) throw new HttpError(404, "document not found");
  return doc;
}

// Save the files, create "queued" documents and hand them to the background jobs (202 Accepted).
router.post("/", upload.array("files"), async (req, res) => {
  if (!req.files?.length) throw new HttpError(422, "no files uploaded (form field 'files')");
  const docs = await Document.insertMany(req.files.map((f) => ({ filename: f.originalname, path: f.path })));
  for (const d of docs) submit(d.id);
  res.status(202).json(docs);
});

router.get("/", async (_req, res) => {
  res.json(await Document.find().sort({ created_at: -1, _id: -1 }));
});

router.get("/:id", async (req, res) => {
  res.json(await findDoc(req.params.id));
});

router.get("/:id/pages", async (req, res) => {
  const doc = await findDoc(req.params.id);
  const pages = await Page.find({ document_id: doc._id }).sort({ number: 1 });
  res.json(pages.map((p) => ({ number: p.number, text: p.text })));
});

router.get("/:id/file", async (req, res) => {
  const doc = await findDoc(req.params.id);
  if (!fs.existsSync(doc.path)) throw new HttpError(404, "original file missing");
  res.download(doc.path, doc.filename);
});

router.delete("/:id", async (req, res) => {
  const doc = await findDoc(req.params.id);
  if (["queued", "processing"].includes(doc.status) || doc.enrich_status === "running") {
    throw new HttpError(409, "document is still being processed");
  }
  await ml.removeDocument(doc.id); // remove from FAISS + BM25 first: no search may return a deleted doc
  fs.rmSync(doc.path, { force: true });
  await Page.deleteMany({ document_id: doc._id });
  await doc.deleteOne();
  res.status(204).end();
});

export default router;
