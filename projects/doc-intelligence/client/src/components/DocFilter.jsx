import { useEffect, useState } from "react";
import { errorMessage, listDocuments } from "../lib/api.js";

// Optional "only search these documents" filter: a checkbox per READY document.
// `selected` is a list of document ids. An empty selection means "all documents"
// (the API treats a missing document_ids the same way).
export default function DocFilter({ selected, onChange }) {
  const [docs, setDocs] = useState([]);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(false);

  useEffect(() => {
    listDocuments()
      .then((all) => setDocs(all.filter((d) => d.status === "ready")))
      .catch((e) => setError(errorMessage(e)));
  }, []);

  function toggle(id) {
    onChange(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id]);
  }

  const summary = selected.length === 0 ? "All documents" : `${selected.length} document(s)`;

  return (
    <div className="relative text-sm">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="rounded border border-border bg-card px-3 py-1.5"
      >
        Filter: {summary} ▾
      </button>
      {open && (
        <div className="absolute left-0 z-10 mt-1 max-h-64 w-72 max-w-[calc(100vw-2rem)] overflow-auto rounded border border-border bg-card p-2 shadow-lg">
          {error && <p className="text-bad">{error}</p>}
          {!error && docs.length === 0 && <p className="text-muted">No ready documents yet.</p>}
          {docs.map((d) => (
            <label key={d.id} className="flex cursor-pointer items-center gap-2 rounded px-1 py-1 hover:bg-bg">
              <input type="checkbox" checked={selected.includes(d.id)} onChange={() => toggle(d.id)} />
              <span className="min-w-0 truncate">{d.filename}</span>
            </label>
          ))}
          {selected.length > 0 && (
            <button type="button" onClick={() => onChange([])} className="mt-1 text-xs text-accent hover:underline">
              Clear (search all)
            </button>
          )}
        </div>
      )}
    </div>
  );
}
