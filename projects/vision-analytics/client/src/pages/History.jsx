import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { listRuns } from "../lib/api";
import { useAsync, useDebounced, usePageTitle } from "../lib/hooks";
import { RunsTable } from "../components/RunsTable";
import { Card, EmptyState, ErrorState, LoadingState, PageHeader } from "../components/ui";

const PAGE_SIZE = 20;

/**
 * GET /api/runs?kind=&label=&q=&limit=&offset=
 * Filters and page live in the URL (?kind=video&q=street&page=2), so a filtered view can be
 * bookmarked/shared and the browser back button works.
 */
export default function History() {
  usePageTitle("History");
  const [sp, setSearchParams] = useSearchParams();
  const kind = sp.get("kind") ?? "";
  const label = sp.get("label") ?? "";
  const q = sp.get("q") ?? "";
  const page = Math.max(1, Number(sp.get("page")) || 1);

  // text inputs update the URL after typing pauses
  const [qInput, setQInput] = useState(q);
  const [labelInput, setLabelInput] = useState(label);
  const dq = useDebounced(qInput, 400);
  const dLabel = useDebounced(labelInput, 400);

  const setParams = (next) => {
    const p = new URLSearchParams(sp.toString());
    for (const [k, v] of Object.entries(next)) {
      if (v === null || v === "" || (k === "page" && v === 1)) p.delete(k);
      else p.set(k, String(v));
    }
    setSearchParams(p, { replace: true, preventScrollReset: true });
  };

  useEffect(() => {
    if (dq.trim() !== q || dLabel.trim() !== label) setParams({ q: dq.trim(), label: dLabel.trim(), page: 1 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dq, dLabel]);

  // keep inputs in sync on back/forward navigation (but not when the change came from typing)
  useEffect(() => {
    if (q !== dq.trim()) setQInput(q);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q]);
  useEffect(() => {
    if (label !== dLabel.trim()) setLabelInput(label);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [label]);

  const runs = useAsync(
    (signal) => listRuns({ kind, label, q, limit: PAGE_SIZE, offset: (page - 1) * PAGE_SIZE }, { signal }),
    [kind, label, q, page],
  );

  const total = runs.data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const hasFilters = !!(kind || label || q);

  return (
    <>
      <PageHeader
        title="History"
        description="Every stored run. Filter by kind, by a detected label, or by file name."
      />

      <Card className="mb-4">
        <form
          className="grid gap-3 sm:grid-cols-[10rem_1fr_1fr_auto] sm:items-end"
          role="search"
          onSubmit={(e) => e.preventDefault()}
        >
          <div>
            <label htmlFor="f-kind" className="label">
              Kind
            </label>
            <select
              id="f-kind"
              className="input"
              value={kind}
              onChange={(e) => setParams({ kind: e.target.value, page: 1 })}
            >
              <option value="">All</option>
              <option value="image">Image</option>
              <option value="video">Video</option>
              <option value="stream">Live stream</option>
            </select>
          </div>
          <div>
            <label htmlFor="f-label" className="label">
              Label
            </label>
            <input
              id="f-label"
              className="input"
              placeholder="e.g. car"
              value={labelInput}
              onChange={(e) => setLabelInput(e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="f-q" className="label">
              File name
            </label>
            <input
              id="f-q"
              type="search"
              className="input"
              placeholder="Search…"
              value={qInput}
              onChange={(e) => setQInput(e.target.value)}
            />
          </div>
          <button
            type="button"
            className="btn"
            disabled={!hasFilters}
            onClick={() => {
              setQInput("");
              setLabelInput("");
              setSearchParams({}, { replace: true, preventScrollReset: true });
            }}
          >
            Clear
          </button>
        </form>
      </Card>

      <Card
        title={runs.data ? `${total} run${total === 1 ? "" : "s"}` : "Runs"}
        actions={
          <button type="button" className="btn" onClick={runs.reload}>
            Refresh
          </button>
        }
      >
        {runs.loading && !runs.data ? (
          <LoadingState />
        ) : runs.error ? (
          <ErrorState message={runs.error} onRetry={runs.reload} />
        ) : runs.data && runs.data.items.length > 0 ? (
          <div className={runs.loading ? "opacity-60 transition-opacity" : ""} aria-busy={runs.loading}>
            <RunsTable runs={runs.data.items} />
          </div>
        ) : (
          <EmptyState>{hasFilters ? "No runs match these filters." : "No runs stored yet."}</EmptyState>
        )}

        {total > PAGE_SIZE && (
          <nav aria-label="Pagination" className="mt-4 flex items-center justify-between gap-2 text-sm">
            <button type="button" className="btn" disabled={page <= 1} onClick={() => setParams({ page: page - 1 })}>
              ← Previous
            </button>
            <span className="text-muted tabular-nums">
              Page {page} of {pages}
            </span>
            <button
              type="button"
              className="btn"
              disabled={page >= pages}
              onClick={() => setParams({ page: page + 1 })}
            >
              Next →
            </button>
          </nav>
        )}
      </Card>
    </>
  );
}
