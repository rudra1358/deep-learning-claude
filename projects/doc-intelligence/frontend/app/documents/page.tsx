"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import UploadZone from "@/components/UploadZone";
import { Empty, ErrorBox, Loading, StatusBadge } from "@/components/ui";
import { deleteDocument, errorMessage, listDocuments, type DocumentOut } from "@/lib/api";

// Documents page: upload + a table of every document and its processing state.
// While anything is still being processed we poll GET /api/documents every 2 s;
// once everything is finished, polling stops by itself (no wasted requests).
export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentOut[] | null>(null);
  const [error, setError] = useState(""); // loading the list failed
  const [deleteError, setDeleteError] = useState(""); // kept separate so the 2 s polling doesn't wipe it

  // useCallback keeps the same function between renders, so the effects below don't re-run needlessly.
  const load = useCallback(async () => {
    try {
      setDocs(await listDocuments());
      setError("");
    } catch (e) {
      setError(errorMessage(e));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Is any document still moving through the pipeline?
  const busy = (docs ?? []).some(
    (d) => d.status === "queued" || d.status === "processing" || d.enrich_status === "running",
  );

  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(load, 2000);
    return () => clearInterval(timer); // stops when busy turns false or the page is left
  }, [busy, load]);

  async function remove(d: DocumentOut) {
    if (!window.confirm(`Delete "${d.filename}"? Its chunks are removed from the search index.`)) return;
    setDeleteError("");
    try {
      await deleteDocument(d.id);
      setDocs((prev) => (prev ?? []).filter((x) => x.id !== d.id));
    } catch (e) {
      setDeleteError(errorMessage(e)); // e.g. 409 while the document is still processing
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Documents</h1>
      <UploadZone onUploaded={load} />

      {error && <ErrorBox message={error} onRetry={load} />}
      {deleteError && <ErrorBox message={deleteError} />}
      {docs === null && !error && <Loading />}
      {docs?.length === 0 && <Empty>No documents yet. Upload one above to get started.</Empty>}

      {docs && docs.length > 0 && (
        // overflow-x-auto: on a phone the table scrolls sideways instead of breaking the layout
        <div className="overflow-x-auto rounded border border-border bg-card">
          <table className="w-full min-w-[720px] text-left text-sm">
            <thead className="border-b border-border text-xs uppercase text-muted">
              <tr>
                <th className="p-3">File</th>
                <th className="p-3">Status</th>
                <th className="p-3 text-right">Pages</th>
                <th className="p-3 text-right">Chunks</th>
                <th className="p-3">Topic</th>
                <th className="p-3">Enrichment</th>
                <th className="p-3">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {docs.map((d) => (
                <tr key={d.id} className="border-b border-border last:border-0">
                  <td className="max-w-[260px] p-3">
                    {d.status === "ready" ? (
                      <Link href={`/documents/${d.id}`} className="break-all text-accent hover:underline">
                        {d.filename}
                      </Link>
                    ) : (
                      <span className="break-all">{d.filename}</span>
                    )}
                    {d.error && <p className="mt-1 text-xs text-bad">{d.error}</p>}
                  </td>
                  <td className="p-3">
                    <StatusBadge status={d.status} />
                  </td>
                  <td className="p-3 text-right">{d.num_pages}</td>
                  <td className="p-3 text-right">{d.num_chunks}</td>
                  <td className="p-3">
                    {d.label ? (
                      <>
                        {d.label}{" "}
                        {d.label_score != null && <span className="text-muted">({(d.label_score * 100).toFixed(0)}%)</span>}
                      </>
                    ) : (
                      <span className="text-muted">—</span>
                    )}
                  </td>
                  <td className="p-3">
                    <StatusBadge status={d.enrich_status} />
                  </td>
                  <td className="p-3 text-right">
                    <button
                      type="button"
                      onClick={() => remove(d)}
                      aria-label={`Delete ${d.filename}`}
                      className="rounded border border-border px-2 py-1 text-xs hover:border-bad hover:text-bad"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {busy && <p className="text-xs text-muted">Refreshing every 2 s while documents are processing…</p>}
    </div>
  );
}
