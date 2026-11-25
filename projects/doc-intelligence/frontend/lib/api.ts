// The ONLY file that talks to the backend.
// Every type below mirrors a Pydantic model in backend/app/schemas.py, and every function maps to
// exactly one endpoint. Pages import from here, so if the API changes there is one place to fix.

// NEXT_PUBLIC_* variables are baked into the browser bundle at build time (the browser calls the API directly).
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8200";

// ---------------------------------------------------------------- types (mirror schemas.py)

export type DocStatus = "queued" | "processing" | "ready" | "failed";
export type EnrichStatus = "pending" | "running" | "done" | "failed";

export interface Entity {
  text: string;
  type: string; // PER | ORG | LOC | MISC (CoNLL-2003 types from our fine-tuned NER model)
  count: number;
}

export interface DocumentOut {
  id: number;
  filename: string;
  status: DocStatus;
  enrich_status: EnrichStatus;
  error: string | null;
  num_pages: number;
  num_chunks: number;
  label: string | null;
  label_score: number | null;
  entities: Entity[] | null;
  summary: string | null;
  created_at: string; // ISO date (UTC)
}

export interface PageOut {
  number: number;
  text: string;
}

export type SearchMode = "hybrid" | "dense" | "bm25";
export type AskMode = "extractive" | "generative";

// Where a chunk ranked in each retriever. null = that retriever did not return it.
// "retrieval" only exists when re-ranking ran: the position BEFORE the cross-encoder re-ordered it.
export interface Ranks {
  dense?: number | null;
  bm25?: number | null;
  retrieval?: number | null;
}

// One retrieved chunk. "n" is its 1-based number, the same number the answer cites as [n].
export interface Source {
  n: number;
  chunk_id: number;
  document_id: number;
  filename: string;
  page_start: number;
  page_end: number;
  char_start: number; // where the chunk begins inside page_start's text
  score: number;
  ranks: Ranks;
  text: string;
}

export interface Timing {
  retrieve_ms?: number;
  rerank_ms?: number;
  answer_ms?: number;
}

// Exact answer span inside sources[source - 1].text (extractive mode only).
export interface Highlight {
  source: number;
  start: number;
  end: number;
}

export interface SearchIn {
  query: string;
  mode: SearchMode;
  rerank: boolean;
  top_k: number;
  document_ids?: number[] | null;
}

export interface SearchOut {
  hits: Source[];
  timing: Timing;
}

export interface AskIn {
  question: string;
  mode: AskMode;
  document_ids?: number[] | null;
  session_id?: number | null;
}

export interface AskOut {
  session_id: number;
  answer: string | null;
  found: boolean;
  mode: AskMode;
  cited: number[];
  confidence?: number | null;
  highlight?: Highlight | null;
  suggestion?: Suggestion | null;
  gated?: boolean; // generative: the extractive reader found no answer, so the LLM was not asked
  sources: Source[];
  timing: Timing;
}

export interface MessageMeta {
  mode?: AskMode;
  found?: boolean;
  cited?: number[] | null;
  highlight?: Highlight | null;
  timing?: Timing;
  confidence?: number | null;
  suggestion?: Suggestion | null;
  gated?: boolean | null;
}

// Extractive mode: the reader's best span when its confidence is above 0 but below the tuned
// threshold. found=false, but the candidate is shown, clearly labelled, so the user can check it.
// confidence / threshold are LOGIT MARGINS (best span score - "no answer" score), not probabilities.
export interface Suggestion {
  text: string;
  confidence: number;
  threshold: number;
  highlight: Highlight;
}

export interface MessageOut {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources: Source[] | null; // null for user messages
  meta: MessageMeta | null;
  created_at: string;
}

export interface SessionOut {
  id: number;
  title: string;
  created_at: string;
}

export interface SessionDetail extends SessionOut {
  messages: MessageOut[];
}

export interface Health {
  status: string;
  device: string;
  embedder: string;
  chunks_indexed: number;
  models_loaded: string[];
  generator_enabled: boolean;
  finetuned_models: Record<string, boolean>;
}

// /api/metrics is free-form JSON written by the training scripts, so we only describe it as "JSON".
export type Json = string | number | boolean | null | Json[] | { [key: string]: Json };
export type Metrics = { [section: string]: Json };

// ---------------------------------------------------------------- the one fetch helper

// An error we can show to the user. status 0 = the request never reached the server.
export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    // fetch only throws when there is no HTTP response at all (server down, CORS, DNS).
    throw new ApiError(0, `Cannot reach the API at ${API_URL}. Is the backend running?`);
  }
  if (!res.ok) {
    // FastAPI puts the reason in {"detail": ...}; validation errors (422) put a list there.
    let detail = res.statusText;
    try {
      const body: unknown = await res.json();
      if (body && typeof body === "object" && "detail" in body) {
        const d = (body as { detail: unknown }).detail;
        detail = typeof d === "string" ? d : JSON.stringify(d);
      }
    } catch {
      /* body was not JSON - keep statusText */
    }
    throw new ApiError(res.status, `${res.status}: ${detail}`);
  }
  if (res.status === 204) return undefined as T; // DELETE endpoints return no body
  return (await res.json()) as T;
}

// Small helper for JSON POST bodies.
function postJson(body: unknown): RequestInit {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

// Turn any thrown value into a readable sentence for the UI.
export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

// ---------------------------------------------------------------- one function per endpoint

// Documents
export function uploadDocuments(files: File[]): Promise<DocumentOut[]> {
  const form = new FormData();
  for (const f of files) form.append("files", f); // same field name repeated = a list on the server
  // No Content-Type header: the browser sets multipart/form-data with the correct boundary itself.
  return request("/api/documents", { method: "POST", body: form });
}
export const listDocuments = () => request<DocumentOut[]>("/api/documents");
export const getDocument = (id: number) => request<DocumentOut>(`/api/documents/${id}`);
export const getPages = (id: number) => request<PageOut[]>(`/api/documents/${id}/pages`);
export const fileUrl = (id: number) => `${API_URL}/api/documents/${id}/file`; // used as a plain <a href>
export const deleteDocument = (id: number) => request<void>(`/api/documents/${id}`, { method: "DELETE" });

// Search and question answering
export const search = (body: SearchIn) => request<SearchOut>("/api/search", postJson(body));
export const ask = (body: AskIn) => request<AskOut>("/api/ask", postJson(body));

// Chat history
export const listChats = () => request<SessionOut[]>("/api/chats");
export const getChat = (id: number) => request<SessionDetail>(`/api/chats/${id}`);
export const deleteChat = (id: number) => request<void>(`/api/chats/${id}`, { method: "DELETE" });

// System
export const getHealth = () => request<Health>("/api/health");
export const getMetrics = () => request<Metrics>("/api/metrics");
