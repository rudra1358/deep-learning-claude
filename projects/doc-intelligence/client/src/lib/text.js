// Small pure helpers (no React, no fetching) shared by several pages. Pure = easy to reason about.

// "p. 3" or "pp. 3–4" when a chunk crosses a page break.
export function pageLabel(s) {
  return s.page_start === s.page_end ? `p. ${s.page_start}` : `pp. ${s.page_start}–${s.page_end}`;
}

// Link from a source card to the document viewer, scrolled to the cited chunk.
// page/start/len locate the chunk; q (the chunk's first words) lets the viewer find it by text,
// which is more robust than raw offsets because the chunker re-joins sentences with single spaces.
export function viewerHref(s) {
  const params = new URLSearchParams({
    page: String(s.page_start),
    start: String(s.char_start),
    len: String(s.text.length),
    q: s.text.slice(0, 80),
  });
  return `/documents/${s.document_id}?${params.toString()}`;
}

// "BM25 #3 · Dense #1 · before rerank #2" - why a chunk was retrieved.
// The API always sends both bm25 and dense keys; in a single-retriever mode the other one is null
// only because it never ran, so we skip it instead of saying "not in top list".
export function ranksLine(r, mode) {
  const fmt = (v) => (v == null ? "not in top list" : `#${v}`);
  const parts = [];
  if (mode !== "dense" && "bm25" in r) parts.push(`BM25 ${fmt(r.bm25)}`);
  if (mode !== "bm25" && "dense" in r) parts.push(`Dense ${fmt(r.dense)}`);
  if (r.retrieval != null) parts.push(`before rerank #${r.retrieval}`);
  return parts.join(" · ");
}

// "retrieve 12 ms · rerank 80 ms · answer 140 ms"
export function timingLine(t) {
  const parts = [];
  if (t.retrieve_ms != null) parts.push(`retrieve ${Math.round(t.retrieve_ms)} ms`);
  if (t.rerank_ms != null) parts.push(`rerank ${Math.round(t.rerank_ms)} ms`);
  if (t.answer_ms != null) parts.push(`answer ${Math.round(t.answer_ms)} ms`);
  return parts.join(" · ");
}

// Find `needle` inside `haystack` while ignoring differences in whitespace (newlines vs spaces).
// Returns the [start, end) position in the ORIGINAL haystack, or null if not found.
// How: collapse every whitespace run to one space, remembering where each kept character came from.
export function findIgnoringSpaces(haystack, needle) {
  let flat = "";
  const origin = []; // origin[i] = index in haystack of flat[i]
  for (let i = 0; i < haystack.length; i++) {
    const isSpace = /\s/.test(haystack[i]);
    if (isSpace && flat.endsWith(" ")) continue;
    flat += isSpace ? " " : haystack[i];
    origin.push(i);
  }
  const target = needle.replace(/\s+/g, " ").trim();
  if (!target) return null;
  const at = flat.indexOf(target);
  if (at < 0) return null;
  const last = origin[at + target.length - 1];
  return [origin[at], last + 1];
}
