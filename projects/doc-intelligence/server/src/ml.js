// The ONLY file that talks to the Python ML service (ml-service/, FastAPI on port 8200).
//
// If the ML service answers with an error, we throw an HttpError with the SAME status and
// message, so the browser sees e.g. "503: generative answers are disabled" exactly as Python said it.
// If it cannot be reached at all, that is a 503 ("service unavailable").
import { Agent, fetch } from "undici";
import { config } from "./config.js";

export class HttpError extends Error {
  constructor(status, detail) {
    super(detail);
    this.status = status;
  }
}

// Model calls can be slow (summarising a long PDF, a 1.5 B LLM on CPU, a model that is still
// downloading). Node's fetch gives up waiting for response headers after 5 minutes by default,
// so we use our own connection pool without that limit; each call sets its own timeout below.
const agent = new Agent({ headersTimeout: 0, bodyTimeout: 0 });

async function call(method, path, body, timeoutMs = 120_000) {
  let res;
  try {
    res = await fetch(`${config.mlUrl}${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(timeoutMs),
      dispatcher: agent,
    });
  } catch (e) {
    throw new HttpError(503, `ML service unreachable at ${config.mlUrl} (${e.cause?.code ?? e.name})`);
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const d = data?.detail ?? res.statusText;
    throw new HttpError(res.status, typeof d === "string" ? d : JSON.stringify(d));
  }
  return data;
}

const MINUTE = 60_000;

export const ml = {
  health: () => call("GET", "/health", null, 5000),
  metrics: () => call("GET", "/metrics"),
  // -> {pages: [{number, text}], num_chunks}
  indexDocument: (doc) =>
    call("POST", "/index/documents", { document_id: doc.id, filename: doc.filename, path: doc.path }, 60 * MINUTE),
  removeDocument: (id) => call("DELETE", `/index/documents/${id}`),
  // -> {label, label_score, entities, summary}
  enrich: (id, text) => call("POST", "/enrich", { document_id: id, text }, 60 * MINUTE),
  search: (body) => call("POST", "/search", body),
  ask: (body) => call("POST", "/ask", body, 10 * MINUTE),
};
