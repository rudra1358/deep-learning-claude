/**
 * Typed client for the Vision Analytics FastAPI backend.
 *
 * The interfaces below mirror backend/app/schemas.py (and the few endpoints that return
 * plain dicts) field-for-field, so a backend change shows up here as a type error.
 * Every function throws an ApiError with the server's `detail` message on non-2xx.
 */

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8100").replace(/\/+$/, "");

/** ws(s)://<same host as API_URL>/api/stream */
export function streamUrl(): string {
  const u = new URL(API_URL);
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  u.pathname = "/api/stream";
  return u.toString();
}

// --------------------------------------------------------------------------- schemas.py

/** [x1, y1, x2, y2] in original-image pixels (backend: list[float], always 4 values). */
export type Box = [number, number, number, number];

/** schemas.DetectionOut */
export interface DetectionOut {
  box: Box;
  score: number;
  cls: number;
  label: string;
}

/** schemas.Timing (milliseconds) */
export interface Timing {
  preprocess_ms: number;
  inference_ms: number;
  postprocess_ms: number;
  total_ms: number;
}

/** schemas.ImageResult */
export interface ImageResult {
  run_id: number | null;
  model: string;
  provider: string;
  width: number;
  height: number;
  detections: DetectionOut[];
  counts: Record<string, number>;
  timing: Timing;
}

/** schemas.CompareResult */
export interface CompareResult {
  results: ImageResult[];
}

export type RunKind = "image" | "video" | "stream";
export type RunStatus = "queued" | "running" | "done" | "failed";

/** In/out crossing counts per label (inference/analytics.py LineCounter.summary). */
export interface LineSummary {
  in: Record<string, number>;
  out: Record<string, number>;
}

/**
 * Run.summary is typed `dict` in schemas.py. Its contents depend on the run kind:
 *  - image:          { counts, timing }                         (routers/detect.py)
 *  - video / stream: { frames, unique_objects, avg_dwell_seconds, line,
 *                      avg_latency_ms, peak_occupancy }         (pipeline.Pipeline.summary)
 * so every key is optional here.
 */
export interface RunSummary {
  counts?: Record<string, number>;
  timing?: Timing;
  frames?: number;
  unique_objects?: Record<string, number>;
  avg_dwell_seconds?: Record<string, number>;
  line?: LineSummary | null;
  avg_latency_ms?: number;
  peak_occupancy?: Record<string, number>;
}

/** schemas.RunOut */
export interface Run {
  id: number;
  kind: RunKind;
  model: string;
  source_name: string;
  status: RunStatus;
  progress: number;
  error: string | null;
  width: number;
  height: number;
  fps: number;
  frames: number;
  avg_latency_ms: number;
  conf: number;
  /** [x1, y1, x2, y2] normalised to 0..1 */
  line: [number, number, number, number] | null;
  summary: RunSummary;
  /** ISO datetime. SQLite drops the timezone, so it may lack a "Z" — use parseServerDate. */
  created_at: string;
}

/** schemas.RunPage */
export interface RunPage {
  total: number;
  items: Run[];
}

export type Direction = "in" | "out";

/** schemas.EventOut */
export interface EventOut {
  track_id: number;
  label: string;
  direction: Direction;
  t_seconds: number;
}

/** schemas.TimeseriesOut — per-second peak count of each class. */
export interface TimeseriesOut {
  labels: string[];
  t: number[];
  series: Record<string, number[]>;
}

/** schemas.OverviewOut */
export interface OverviewOut {
  runs_by_kind: Record<string, number>;
  objects_by_label: Record<string, number>;
  /** {"in": {label: n}, "out": {label: n}} */
  crossings: Record<string, Record<string, number>>;
  avg_latency_ms: number | null;
}

// --------------------------------------------------------------------------- plain-dict endpoints

/** GET /api/health (main.py) */
export interface Health {
  status: string;
  providers: string[];
  models: string[];
  default_model: string;
}

/** GET /api/models item (models_registry.describe). provider/imgsz/classes only when loaded. */
export interface ModelInfo {
  name: string;
  loaded: boolean;
  metrics: ModelExportMetrics | null;
  provider?: string;
  imgsz?: number;
  /** JSON object keys are strings: {"0": "pedestrian", ...} */
  classes?: Record<string, string>;
}

/** GET /api/runs/{id}/detections item (image runs) */
export interface StoredDetection {
  label: string;
  score: number;
  box: Box;
}

/** One tracked object in a frame (pipeline.Pipeline.process). */
export interface Track {
  id: number;
  label: string;
  score: number;
  box: Box;
  /** last ≤15 centre points in pixels */
  trail: [number, number][];
}

/** GET /api/runs/{id}/frames — data/runs/<id>/frames.json written by video_jobs.py */
export interface FramesFile {
  fps: number;
  width: number;
  height: number;
  frames: { t: number; tracks: Track[] }[];
}

// --------------------------------------------------------------------------- /api/metrics
// Written by training/evaluate.py and training/export_onnx.py via common.update_metrics.

export interface CocoEval {
  mAP50_95: number;
  mAP50: number;
  AP_small?: number;
  AP_medium?: number;
  AP_large?: number;
}

export interface CocoBaselineEntry extends CocoEval {
  latency_ms?: number;
  params_M?: number;
}

export interface FinetunedEntry {
  mAP50_95: number;
  mAP50: number;
  precision?: number;
  recall?: number;
  per_class_AP50_95?: Record<string, number>;
  by_size?: CocoEval;
  shared_classes_for_domain_gap?: CocoEval;
  confusion_matrix_png?: string;
}

export interface ParityResult {
  reference_boxes: number;
  onnx_boxes: number;
  match_rate: number;
  mean_iou_of_matches: number | null;
  max_score_diff: number | null;
}

export interface ModelExportMetrics {
  onnx_mb?: number;
  parity?: ParityResult;
  int8_parity?: ParityResult;
  latency_ms?: {
    cpu_fp32?: number | null;
    cpu_int8?: number | null;
    cuda_fp32?: number | null;
    cuda_fp16?: number | null;
  };
}

export interface MetricsFile {
  coco_baseline?: Record<string, CocoBaselineEntry>;
  domain_gap?: Record<string, CocoEval>;
  finetuned?: Record<string, FinetunedEntry>;
  models?: Record<string, ModelExportMetrics>;
}

// --------------------------------------------------------------------------- /api/stream (WebSocket)

export interface StreamConfigMsg {
  type: "config";
  model: string | null;
  conf: number;
  line: [number, number, number, number] | null;
  width: number;
  height: number;
}

export interface CrossingEvent {
  track_id: number;
  label: string;
  direction: Direction;
}

export type StreamServerMsg =
  | { type: "ready"; model: string; provider: string }
  | {
      type: "result";
      frame: number;
      tracks: Track[];
      occupancy: Record<string, number>;
      events: CrossingEvent[];
      timing: Timing;
      server_fps: number;
    }
  | { type: "saved"; run_id: number; summary: RunSummary }
  | { type: "saved"; run_id: null; summary: null } // stop before any frame was processed: nothing saved
  | { type: "error"; message: string };

export type StreamResultMsg = Extract<StreamServerMsg, { type: "result" }>;

// --------------------------------------------------------------------------- HTTP helpers

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** FastAPI errors are {"detail": "..."} or {"detail": [{"msg": "...", ...}]} for validation. */
function detailMessage(body: unknown): string | null {
  if (typeof body !== "object" || body === null || !("detail" in body)) return null;
  const d = (body as { detail: unknown }).detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) {
    return d
      .map((e) => (typeof e === "object" && e !== null && "msg" in e ? String((e as { msg: unknown }).msg) : String(e)))
      .join("; ");
  }
  return null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(`Cannot reach the API at ${API_URL}. Is the backend running?`, 0);
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
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

type Opts = { signal?: AbortSignal };

// --------------------------------------------------------------------------- endpoints

export const getHealth = (o?: Opts) => request<Health>("/api/health", o);
export const getModels = (o?: Opts) => request<ModelInfo[]>("/api/models", o);
export const getMetrics = (o?: Opts) => request<MetricsFile>("/api/metrics", o);
export const getOverview = (o?: Opts) => request<OverviewOut>("/api/analytics/overview", o);

export interface RunFilters {
  kind?: string;
  label?: string;
  q?: string;
  limit?: number;
  offset?: number;
}

export function listRuns(f: RunFilters = {}, o?: Opts): Promise<RunPage> {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(f)) {
    if (v !== undefined && v !== "") p.set(k, String(v));
  }
  const qs = p.toString();
  return request<RunPage>(`/api/runs${qs ? `?${qs}` : ""}`, o);
}

export const getRun = (id: number, o?: Opts) => request<Run>(`/api/runs/${id}`, o);
export const deleteRun = (id: number) => request<void>(`/api/runs/${id}`, { method: "DELETE" });
export const getRunDetections = (id: number, o?: Opts) => request<StoredDetection[]>(`/api/runs/${id}/detections`, o);
export const getRunFrames = (id: number, o?: Opts) => request<FramesFile>(`/api/runs/${id}/frames`, o);
export const getRunEvents = (id: number, o?: Opts) => request<EventOut[]>(`/api/runs/${id}/events`, o);
export const getRunTimeseries = (id: number, o?: Opts) => request<TimeseriesOut>(`/api/runs/${id}/timeseries`, o);
export const runVideoUrl = (id: number) => `${API_URL}/api/runs/${id}/video`;
export const exportCsvUrl = (id: number) => `${API_URL}/api/runs/${id}/export.csv`;

export interface DetectImageParams {
  file: Blob;
  filename?: string;
  model?: string | null;
  conf: number;
  iou: number;
  save: boolean;
}

/** POST /api/detect/image (multipart). */
export function detectImage(p: DetectImageParams, o?: Opts): Promise<ImageResult> {
  const fd = new FormData();
  fd.append("file", p.file, p.filename ?? (p.file instanceof File ? p.file.name : "image.jpg"));
  if (p.model) fd.append("model", p.model);
  fd.append("conf", String(p.conf));
  fd.append("iou", String(p.iou));
  fd.append("save", String(p.save));
  return request<ImageResult>("/api/detect/image", { method: "POST", body: fd, signal: o?.signal });
}

/** POST /api/compare — models are sent comma-separated: "a,b". */
export function compareModels(
  p: { file: File; models: string[]; conf: number; iou: number },
  o?: Opts,
): Promise<CompareResult> {
  const fd = new FormData();
  fd.append("file", p.file, p.file.name);
  fd.append("models", p.models.join(","));
  fd.append("conf", String(p.conf));
  fd.append("iou", String(p.iou));
  return request<CompareResult>("/api/compare", { method: "POST", body: fd, signal: o?.signal });
}

/** POST /api/detect/video — returns the queued run (HTTP 202). `line` is sent as a JSON string. */
export function detectVideo(p: {
  file: File;
  model?: string | null;
  conf: number;
  line: [number, number, number, number] | null;
}): Promise<Run> {
  const fd = new FormData();
  fd.append("file", p.file, p.file.name);
  if (p.model) fd.append("model", p.model);
  fd.append("conf", String(p.conf));
  if (p.line) fd.append("line", JSON.stringify(p.line));
  return request<Run>("/api/detect/video", { method: "POST", body: fd });
}

// --------------------------------------------------------------------------- misc

/** Backend datetimes may come without a timezone (SQLite); they are UTC. */
export function parseServerDate(s: string): Date {
  return new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(s) ? s : `${s}Z`);
}

export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}
