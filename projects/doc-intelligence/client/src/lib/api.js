// The ONLY file that talks to the server.
// Every function maps to exactly one endpoint of the Express server (server/src/routes/*.js).
// Pages import from here, so if the API changes there is one place to fix.
//
// The data shapes (what each endpoint returns), for reference:
//
//   Document  {id, filename, status: "queued"|"processing"|"ready"|"failed",
//              enrich_status: "pending"|"running"|"done"|"failed"|"skipped", error, num_pages, num_chunks,
//              label, label_score, entities: [{text, type: PER|ORG|LOC|MISC, count}] | null, summary, created_at}
//   Page      {number, text}
//   Source    one retrieved chunk: {n, chunk_id, document_id, filename, page_start, page_end,
//              char_start (where the chunk begins inside page_start's text), score, ranks, text}
//              "n" is its 1-based number, the same number a generative answer cites as [n].
//              ranks = where it ranked in each retriever {dense?, bm25?, retrieval?}; null = that retriever
//              did not return it; "retrieval" = its position BEFORE the cross-encoder re-ordered it.
//   Highlight {source, start, end}: exact answer span inside sources[source - 1].text
//   Suggestion {text, confidence, threshold, highlight}: extractive mode, the reader's best span when its
//              confidence is above 0 but below the tuned threshold. confidence / threshold are LOGIT
//              MARGINS (best span score - "no answer" score), not probabilities.
//   AskOut    {session_id, answer, found, mode, cited, confidence, highlight, suggestion,
//              gated (generative: the reader found no answer, so the LLM was not asked), sources, timing}
//   Session   {id, title, created_at}  -  with messages: [{id, role, content, sources, meta, created_at}]
//   Health    {status, mongo, ml, device, embedder, chunks_indexed, models_loaded, generator_enabled, ...}
//
// Ids are MongoDB ObjectIds: 24-character strings like "665f1c2e9b1d4a0012345678".

// Same origin by default: in development Vite proxies /api to the server, in production the server
// serves this app itself. VITE_API_URL (read at build time) points the UI at another server instead.
export const API_URL = import.meta.env.VITE_API_URL ?? "";

// ---------------------------------------------------------------- the one fetch helper

// An error we can show to the user. status 0 = the request never reached the server.
export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function request(path, init) {
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    // fetch only throws when there is no HTTP response at all (server down, CORS, DNS).
    throw new ApiError(0, `Cannot reach the API${API_URL ? ` at ${API_URL}` : ""}. Is the server running?`);
  }
  if (!res.ok) {
    // The server puts the reason in {"detail": ...} (the ML service's messages are passed through as-is).
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body && typeof body === "object" && "detail" in body) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      /* body was not JSON - keep statusText */
    }
    throw new ApiError(res.status, `${res.status}: ${detail}`);
  }
  if (res.status === 204) return undefined; // DELETE endpoints return no body
  return res.json();
}

// Small helper for JSON POST bodies.
function postJson(body) {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

// Turn any thrown value into a readable sentence for the UI.
export function errorMessage(e) {
  return e instanceof Error ? e.message : String(e);
}

// ---------------------------------------------------------------- one function per endpoint

// Documents
export function uploadDocuments(files) {
  const form = new FormData();
  for (const f of files) form.append("files", f); // same field name repeated = a list on the server
  // No Content-Type header: the browser sets multipart/form-data with the correct boundary itself.
  return request("/api/documents", { method: "POST", body: form });
}
export const listDocuments = () => request("/api/documents");
export const getDocument = (id) => request(`/api/documents/${id}`);
export const getPages = (id) => request(`/api/documents/${id}/pages`);
export const fileUrl = (id) => `${API_URL}/api/documents/${id}/file`; // used as a plain <a href>
export const deleteDocument = (id) => request(`/api/documents/${id}`, { method: "DELETE" });

// Search and question answering
// search({query, mode: "hybrid"|"dense"|"bm25", rerank, top_k, document_ids}) -> {hits: [Source], timing}
export const search = (body) => request("/api/search", postJson(body));
// ask({question, mode: "extractive"|"generative", document_ids, session_id}) -> AskOut
export const ask = (body) => request("/api/ask", postJson(body));

// Chat history
export const listChats = () => request("/api/chats");
export const getChat = (id) => request(`/api/chats/${id}`);
export const deleteChat = (id) => request(`/api/chats/${id}`, { method: "DELETE" });

// System
export const getHealth = () => request("/api/health");
// /api/metrics is free-form JSON written by the training scripts.
export const getMetrics = () => request("/api/metrics");
