// The Express application (routes + middleware). index.js starts it; tests import it directly.
import fs from "node:fs";
import path from "node:path";
import cors from "cors";
import express from "express";
import { config } from "./config.js";
import { errorHandler, HttpError } from "./errors.js";
import { detectRouter } from "./routes/detect.js";
import { runsRouter } from "./routes/runs.js";
import { systemRouter } from "./routes/system.js";

export function createApp() {
  const app = express();
  app.use(cors()); // dev: the Vite server on :3100 may call us directly (normally it proxies /api)
  app.use(express.json({ limit: "1mb" }));

  app.use("/api", systemRouter);
  app.use("/api", detectRouter);
  app.use("/api", runsRouter);
  app.use("/api", (_req, _res) => {
    throw new HttpError(404, "not found");
  });

  // Production: serve the built React app; any non-API path returns index.html so that
  // React Router can handle URLs like /runs/123 after a page refresh.
  if (fs.existsSync(path.join(config.clientDist, "index.html"))) {
    app.use(express.static(config.clientDist));
    app.get("/{*path}", (_req, res) => res.sendFile(path.join(config.clientDist, "index.html")));
  }

  app.use(errorHandler);
  return app;
}
