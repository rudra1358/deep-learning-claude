// The Express application: middleware, routes, error handling. No listen() here, so tests can
// use the app directly (supertest) without opening a port.
import fs from "node:fs";
import path from "node:path";
import cors from "cors";
import express from "express";
import multer from "multer";
import { config } from "./config.js";
import documents from "./routes/documents.js";
import qa from "./routes/qa.js";
import system from "./routes/system.js";

export function createApp() {
  const app = express();
  app.disable("x-powered-by");
  app.use(cors({ origin: config.corsOrigins }));
  app.use(express.json({ limit: "1mb" }));

  app.use("/api/documents", documents);
  app.use("/api", qa);
  app.use("/api", system);
  app.use("/api", (_req, _res, next) => next(Object.assign(new Error("Not Found"), { status: 404 })));

  // Production: serve the built React app too, so the whole UI + API is one origin on one port.
  // Any non-API GET returns index.html and React Router shows the right page (SPA fallback).
  if (fs.existsSync(path.join(config.clientDist, "index.html"))) {
    app.use(express.static(config.clientDist));
    app.use((req, res, next) => {
      if (req.method !== "GET") return next();
      res.sendFile(path.join(config.clientDist, "index.html"));
    });
  }

  // Express 5 sends errors thrown in async handlers here automatically.
  // Every error leaves as {"detail": "..."} - the same shape FastAPI uses.
  app.use((err, _req, res, _next) => {
    let status = err.status ?? err.statusCode ?? 500;
    if (err instanceof multer.MulterError) {
      status = err.code === "LIMIT_FILE_SIZE" ? 413 : 400;
      if (status === 413) err.message = `file larger than ${config.maxUploadMb} MB`;
    }
    if (status >= 500 && status !== 503) console.error(err);
    res.status(status).json({ detail: status === 500 ? "internal server error" : err.message });
  });
  return app;
}
