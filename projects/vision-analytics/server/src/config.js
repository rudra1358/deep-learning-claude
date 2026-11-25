// All settings in one place, read from environment variables (or a .env file in server/).
import path from "node:path";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";

dotenv.config({ quiet: true });

const here = path.dirname(fileURLToPath(import.meta.url)); // .../server/src
const projectRoot = path.resolve(here, "..", "..");        // .../vision-analytics

export const config = {
  port: Number(process.env.PORT ?? 4100),
  mongoUrl: process.env.MONGO_URL ?? "mongodb://127.0.0.1:27017/vision",
  // The Python ML service (ml-service/). Only this server talks to it - never the browser.
  mlUrl: (process.env.ML_URL ?? "http://127.0.0.1:8100").replace(/\/+$/, ""),
  // Shared with the ML service: we save uploads in DATA_DIR/uploads, it writes DATA_DIR/runs/<id>/.
  dataDir: path.resolve(process.env.DATA_DIR ?? path.join(projectRoot, "data")),
  maxUploadMb: Number(process.env.MAX_UPLOAD_MB ?? 200),
  // Production: the built React app (client/dist) is served by this server, so one URL serves everything.
  clientDist: path.resolve(process.env.CLIENT_DIST ?? path.join(projectRoot, "client", "dist")),
  // How often the server asks the ML service how a video job is going.
  jobPollMs: Number(process.env.JOB_POLL_MS ?? 1000),
};
