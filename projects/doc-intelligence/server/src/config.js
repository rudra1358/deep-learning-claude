// All settings in one place, read from environment variables (or a .env file next to package.json).
// Paths are resolved from the server/ folder, so `npm start` works from any working directory.
import path from "node:path";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";

const SERVER_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
dotenv.config({ path: path.join(SERVER_DIR, ".env"), quiet: true });

const env = process.env;

export const config = {
  port: Number(env.PORT ?? 4200),
  mongoUrl: env.MONGO_URL ?? "mongodb://127.0.0.1:27017/docintel",
  mlUrl: (env.ML_URL ?? "http://127.0.0.1:8200").replace(/\/$/, ""), // the Python ML service
  // Uploads are saved here. The ML service must see the SAME folder (it reads the file to index it):
  // locally both use projects/doc-intelligence/data, in Docker both mount one volume at /app/data.
  dataDir: path.resolve(SERVER_DIR, env.DATA_DIR ?? "../data"),
  maxUploadMb: Number(env.MAX_UPLOAD_MB ?? 50),
  // The built React app (client/dist). If the folder exists, the server also serves the UI.
  clientDist: path.resolve(SERVER_DIR, env.CLIENT_DIST ?? "../client/dist"),
  // Only needed when the UI is served from another origin (e.g. Vite dev without its proxy).
  corsOrigins: (env.CORS_ORIGINS ?? "http://localhost:3200,http://127.0.0.1:3200").split(","),
};

// File types the ML service can extract text from (ml-service/ingest/extract.py SUPPORTED).
export const SUPPORTED = [".docx", ".md", ".pdf", ".txt"];
