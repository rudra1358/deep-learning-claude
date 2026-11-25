// Health, model list and offline metrics. Models and metrics come from the ML service.
import { Router } from "express";
import { dbState } from "../db.js";
import { mlRequest } from "../ml.js";

export const systemRouter = Router();

// Health of the whole stack: this server, MongoDB and the ML service.
systemRouter.get("/health", async (_req, res) => {
  let ml = null;
  try {
    ml = await mlRequest("/health", { timeoutMs: 5000 });
  } catch {
    /* reported below */
  }
  const mongo = dbState();
  res.json({
    status: mongo === "ok" && ml ? "ok" : "degraded",
    mongo,
    ml: ml ? "ok" : "down",
    providers: ml?.providers ?? [],
    models: ml?.models ?? [],
    default_model: ml?.default_model ?? null,
  });
});

systemRouter.get("/models", async (_req, res) => {
  res.json(await mlRequest("/models"));
});

systemRouter.get("/metrics", async (_req, res) => {
  res.json(await mlRequest("/metrics"));
});
