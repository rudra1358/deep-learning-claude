import { useState } from "react";
import DocFilter from "../components/DocFilter.jsx";
import SourceCard from "../components/SourceCard.jsx";
import { Empty, ErrorBox, Loading } from "../components/ui.jsx";
import { errorMessage, search } from "../lib/api.js";
import { ranksLine, timingLine } from "../lib/text.js";

// One sentence per retrieval mode, shown in the UI so a visitor understands the choice.
const MODES = [
  { value: "hybrid", label: "Hybrid", help: "Runs both keyword and semantic search and merges the two rankings (Reciprocal Rank Fusion)." },
  { value: "dense", label: "Dense (semantic)", help: "Embeds the query and finds chunks with similar meaning, even when the words differ." },
  { value: "bm25", label: "BM25 (keyword)", help: "Classic keyword matching: rewards rare words that appear in both the query and the chunk." },
];

// Search page: raw retrieval without answering - useful to SEE why a chunk was found.
export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [rerank, setRerank] = useState(true);
  const [topK, setTopK] = useState(5);
  const [docIds, setDocIds] = useState([]);
  const [result, setResult] = useState(null); // {hits, timing}
  const [resultMode, setResultMode] = useState("hybrid"); // mode the shown results came from
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submit(e) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError("");
    try {
      setResult(await search({ query: query.trim(), mode, rerank, top_k: topK, document_ids: docIds.length ? docIds : null }));
      setResultMode(mode);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Search</h1>

      <form onSubmit={submit} className="space-y-3 rounded border border-border bg-card p-4">
        <div className="flex flex-col gap-2 sm:flex-row">
          <label htmlFor="query" className="sr-only">
            Search query
          </label>
          <input
            id="query"
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search your documents…"
            maxLength={1000}
            className="min-w-0 flex-1 rounded border border-border bg-bg px-3 py-2"
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="rounded bg-accent px-4 py-2 text-sm font-medium text-accent-fg disabled:opacity-50"
          >
            {loading ? "Searching…" : "Search"}
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-4 text-sm">
          <label className="flex items-center gap-2">
            Mode
            <select value={mode} onChange={(e) => setMode(e.target.value)} className="rounded border border-border bg-bg px-2 py-1">
              {MODES.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} />
            Re-rank with cross-encoder
          </label>
          <label className="flex items-center gap-2">
            Top k
            <input
              type="number"
              min={1}
              max={50}
              value={topK}
              // clamp to the range the API accepts (1..50)
              onChange={(e) => setTopK(Math.min(50, Math.max(1, Number(e.target.value) || 1)))}
              className="w-16 rounded border border-border bg-bg px-2 py-1"
            />
          </label>
          <DocFilter selected={docIds} onChange={setDocIds} />
        </div>

        <ul className="space-y-1 text-xs text-muted">
          {MODES.map((m) => (
            <li key={m.value} className={m.value === mode ? "text-fg" : ""}>
              <b>{m.label}:</b> {m.help}
            </li>
          ))}
          <li className={rerank ? "text-fg" : ""}>
            <b>Re-rank:</b> a cross-encoder reads the query and each of the top 30 candidates together and re-orders
            them - slower but more accurate.
          </li>
        </ul>
      </form>

      {error && <ErrorBox message={error} />}
      {loading && <Loading text="Searching…" />}
      {!loading && result && result.hits.length === 0 && <Empty>No matching chunks. Try other words or another mode.</Empty>}
      {!loading && result && result.hits.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs text-muted">
            {result.hits.length} results · {timingLine(result.timing)}
          </p>
          {result.hits.map((h) => (
            <SourceCard key={h.chunk_id} source={h} why={ranksLine(h.ranks, resultMode)} />
          ))}
        </div>
      )}
    </div>
  );
}
