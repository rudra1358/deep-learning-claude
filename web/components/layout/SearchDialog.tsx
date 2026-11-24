"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Search } from "lucide-react";
import { searchDocs, type SearchDoc } from "@/lib/search";

let cache: SearchDoc[] | null = null;

export function SearchDialog({ onClose }: { onClose: () => void }) {
  const [docs, setDocs] = useState<SearchDoc[] | null>(cache);
  const [q, setQ] = useState("");
  const [active, setActive] = useState(0);
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
    if (!cache)
      fetch("/search-index.json")
        .then((r) => r.json())
        .then((d: SearchDoc[]) => {
          cache = d;
          setDocs(d);
        })
        .catch(() => setDocs([]));
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const hits = useMemo(() => (docs ? searchDocs(docs, q, 25) : []), [docs, q]);
  useEffect(() => setActive(0), [q]);

  const go = (href: string) => {
    onClose();
    router.push(href);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/30 px-3 pt-[10vh]" onClick={onClose} role="dialog" aria-modal aria-label="Search">
      <div className="card w-full max-w-2xl overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-2 border-b border-border px-4">
          <Search size={17} className="text-faint" />
          <input
            ref={inputRef}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") {
                e.preventDefault();
                setActive((a) => Math.min(a + 1, hits.length - 1));
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setActive((a) => Math.max(a - 1, 0));
              } else if (e.key === "Enter" && hits[active]) go(hits[active].doc.href);
            }}
            placeholder="Search topics, formulas, code, glossary…"
            className="h-12 w-full bg-transparent text-[0.95rem] outline-none placeholder:text-faint"
            aria-label="Search query"
          />
          <kbd className="rounded border border-border px-1.5 font-mono text-[0.65rem] text-faint">Esc</kbd>
        </div>
        <div className="max-h-[60vh] overflow-y-auto">
          {!docs && <div className="p-6 text-sm text-faint">Loading index…</div>}
          {docs && q.length >= 2 && hits.length === 0 && <div className="p-6 text-sm text-faint">No results for “{q}”.</div>}
          {docs && q.length < 2 && <div className="p-6 text-sm text-faint">Try “backpropagation”, “softmax”, “stride”, “np.einsum” or “KL divergence”.</div>}
          <ul>
            {hits.map((h, i) => (
              <li key={h.doc.href + h.doc.title}>
                <button onClick={() => go(h.doc.href)} onMouseEnter={() => setActive(i)} className={`block w-full px-4 py-2.5 text-left ${i === active ? "bg-surface-2" : ""}`}>
                  <div className="flex items-center gap-2">
                    <span className="rounded bg-accent-soft px-1.5 py-0.5 text-[0.65rem] uppercase tracking-wider text-accent-strong">{h.doc.kind}</span>
                    <span className="text-sm font-medium">{h.doc.title}</span>
                  </div>
                  <div className="mt-0.5 truncate text-xs text-faint">
                    {h.doc.context} — {h.snippet}
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
