// Small HTTP client for the Python ML service (ml-service/).
//
// If the ML service answers with an error, we pass its status code and message straight on
// (e.g. 404 "model not found", 415 "unsupported image type"). If it cannot be reached at all,
// the browser gets 503 Service Unavailable.
import { config } from "./config.js";
import { HttpError } from "./errors.js";

export async function mlRequest(path, { method = "GET", json, form, timeoutMs = 120_000 } = {}) {
  const init = { method, signal: AbortSignal.timeout(timeoutMs) };
  if (json !== undefined) {
    init.headers = { "content-type": "application/json" };
    init.body = JSON.stringify(json);
  } else if (form !== undefined) {
    init.body = form; // FormData: fetch sets the multipart content-type + boundary itself
  }
  let res;
  try {
    res = await fetch(`${config.mlUrl}${path}`, init);
  } catch {
    throw new HttpError(503, `ML service unavailable at ${config.mlUrl}`);
  }
  if (res.status === 204) return null;
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new HttpError(res.status, detailMessage(body) ?? `ML service error ${res.status}`);
  return body;
}

// FastAPI errors are {"detail": "..."} or, for validation errors, {"detail": [{"msg": "..."}, ...]}.
function detailMessage(body) {
  const d = body?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((e) => e?.msg ?? String(e)).join("; ");
  return null;
}

// Build a multipart form for the ML service from a file multer kept in memory.
export function fileForm(file, fields = {}) {
  const form = new FormData();
  form.append("file", new Blob([file.buffer], { type: file.mimetype }), file.originalname);
  for (const [k, v] of Object.entries(fields)) {
    if (v !== undefined && v !== null && v !== "") form.append(k, String(v));
  }
  return form;
}
