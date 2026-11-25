// Health and offline evaluation metrics.  Mounted at /api.
import { Router } from "express";
import { dbIsUp } from "../db.js";
import { ml } from "../ml.js";

const router = Router();

// One status for the whole stack: MongoDB (this server's database) + the ML service's own report.
// Always 200, so the UI can show WHICH part is down instead of a generic error.
router.get("/health", async (_req, res) => {
  const mongo = dbIsUp() ? "ok" : "down";
  try {
    const m = await ml.health();
    res.json({ ...m, status: mongo === "ok" ? "ok" : "degraded", mongo, ml: "ok" });
  } catch (e) {
    res.json({ status: "degraded", mongo, ml: "down", ml_error: e.message });
  }
});

// Results written by the training/evaluation scripts (training/reports/metrics.json), served by Python.
router.get("/metrics", async (_req, res) => {
  res.json(await ml.metrics());
});

export default router;
