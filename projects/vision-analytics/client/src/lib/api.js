/**
 * Client for the Vision Analytics Express API (server/).
 *
 * The browser only ever talks to the Express server - never to the Python ML service.
 * In development the Vite dev server forwards /api/* to Express (see vite.config.js), and in
 * production Express serves this app itself, so relative URLs ("/api/...") work in both cases.
 * Set VITE_API_URL only if the API lives on a different host.
 *
 * Every function throws an ApiError with the server's `detail` message on non-2xx.
 *
 * Main shapes (ids are MongoDB ObjectId strings):
 *   Run         {id, kind: "image"|"video"|"stream", model, source_name, status: "queued"|"running"|"done"|"failed",
 *                progress, error, width, height, fps, frames, avg_latency_ms, conf, line: [x1,y1,x2,y2]|null,
 *                summary, created_at}
 *   ImageResult {run_id, model, provider, width, height, detections: [{box, score, cls, label}], counts, timing}
 *   RunPage     {total, items: Run[]}
 *   Track       {id, label, score, box: [x1,y1,x2,y2], trail: [[x,y], ...]}     (frames.json and live results)
 */

export const API_URL = (import.meta.env.VITE_API_URL ?? "").replace(/\/+$/, "");

/** ws(s)://<API host>/api/stream - the Express server relays it to the ML service. */
export function streamUrl() {
  const u = new URL(API_URL || window.location.origin);
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  u.pathname = "/api/stream";
  return u.toString();
}

// --------------------------------------------------------------------------- HTTP helpers

export class ApiError extends Error {
  status;
  constructor(message, status) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

/** Errors are {"detail": "..."} (or {"detail": [{"msg": "...", ...}]} for validation errors). */
function detailMessage(body) {
  if (typeof body !== "object" || body === null || !("detail" in body)) return null;
  const d = body.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) {
    return d.map((e) => (typeof e === "object" && e !== null && "msg" in e ? String(e.msg) : String(e))).join("; ");
  }
  return null;
}

async function request(path, init) {
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(`Cannot reach the API at ${API_URL || window.location.origin}. Is the server running?`, 0);
  }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      msg = detailMessage(await res.json()) ?? msg;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(msg, res.status);
  }
  if (res.status === 204) return undefined;
  return await res.json();
}

// --------------------------------------------------------------------------- endpoints

export const getHealth = (o) => request("/api/health", o);
export const getModels = (o) => request("/api/models", o);
export const getMetrics = (o) => request("/api/metrics", o);
export const getOverview = (o) => request("/api/analytics/overview", o);

export function listRuns(f = {}, o) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(f)) {
    if (v !== undefined && v !== "") p.set(k, String(v));
  }
  const qs = p.toString();
  return request(`/api/runs${qs ? `?${qs}` : ""}`, o);
}

export const getRun = (id, o) => request(`/api/runs/${id}`, o);
export const deleteRun = (id) => request(`/api/runs/${id}`, { method: "DELETE" });
export const getRunDetections = (id, o) => request(`/api/runs/${id}/detections`, o);
export const getRunFrames = (id, o) => request(`/api/runs/${id}/frames`, o);
export const getRunEvents = (id, o) => request(`/api/runs/${id}/events`, o);
export const getRunTimeseries = (id, o) => request(`/api/runs/${id}/timeseries`, o);
export const runVideoUrl = (id) => `${API_URL}/api/runs/${id}/video`;
export const exportCsvUrl = (id) => `${API_URL}/api/runs/${id}/export.csv`;

/** POST /api/detect/image (multipart). */
export function detectImage(p, o) {
  const fd = new FormData();
  fd.append("file", p.file, p.filename ?? (p.file instanceof File ? p.file.name : "image.jpg"));
  if (p.model) fd.append("model", p.model);
  fd.append("conf", String(p.conf));
  fd.append("iou", String(p.iou));
  fd.append("save", String(p.save));
  return request("/api/detect/image", { method: "POST", body: fd, signal: o?.signal });
}

/** POST /api/compare — models are sent comma-separated: "a,b". */
export function compareModels(p, o) {
  const fd = new FormData();
  fd.append("file", p.file, p.file.name);
  fd.append("models", p.models.join(","));
  fd.append("conf", String(p.conf));
  fd.append("iou", String(p.iou));
  return request("/api/compare", { method: "POST", body: fd, signal: o?.signal });
}

/** POST /api/detect/video — returns the queued run (HTTP 202). `line` is sent as a JSON string. */
export function detectVideo(p) {
  const fd = new FormData();
  fd.append("file", p.file, p.file.name);
  if (p.model) fd.append("model", p.model);
  fd.append("conf", String(p.conf));
  if (p.line) fd.append("line", JSON.stringify(p.line));
  return request("/api/detect/video", { method: "POST", body: fd });
}

// --------------------------------------------------------------------------- misc

/** Server datetimes are ISO strings in UTC ("2026-11-25T10:00:00.000Z"). */
export function parseServerDate(s) {
  return new Date(s);
}

export function errorMessage(err) {
  if (err instanceof Error) return err.message;
  return String(err);
}
