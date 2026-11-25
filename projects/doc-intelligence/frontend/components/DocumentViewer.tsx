"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { ErrorBox, Loading, StatusBadge } from "@/components/ui";
import { errorMessage, fileUrl, getDocument, getPages, type DocumentOut, type Entity, type PageOut } from "@/lib/api";
import { findIgnoringSpaces } from "@/lib/text";

// Which chunk to highlight, taken from the URL (all optional).
export interface Target {
  page: number | null;
  start: number | null;
  len: number | null;
  q: string | null;
}

const ENTITY_TYPES: Record<string, string> = { PER: "People", ORG: "Organisations", LOC: "Places", MISC: "Other" };

// Work out [start, end) of the cited chunk inside the page text.
// 1) search for the chunk's first words (robust)  2) otherwise trust the char offset from the API.
function locate(page: PageOut, t: Target): [number, number] | null {
  const found = t.q ? findIgnoringSpaces(page.text, t.q) : null;
  const start = found ? found[0] : t.start;
  if (start == null || start < 0 || start >= page.text.length) return null;
  const len = t.len ?? (found ? found[1] - found[0] : 200);
  return [start, Math.min(page.text.length, start + len)]; // a chunk can run onto the next page: clip
}

export default function DocumentViewer({ id, target }: { id: number; target: Target }) {
  const [doc, setDoc] = useState<DocumentOut | null>(null);
  const [pages, setPages] = useState<PageOut[] | null>(null);
  const [error, setError] = useState("");
  const markRef = useRef<HTMLElement>(null);

  useEffect(() => {
    // Load the document info and its pages in parallel.
    Promise.all([getDocument(id), getPages(id)])
      .then(([d, p]) => {
        setDoc(d);
        setPages(p);
      })
      .catch((e) => setError(errorMessage(e)));
  }, [id]);

  // Once the pages are on screen, scroll to the highlight (or to the requested page).
  useEffect(() => {
    if (!pages) return;
    const el = markRef.current ?? document.getElementById(`page-${target.page}`);
    el?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [pages, target.page]);

  if (error) return <ErrorBox message={error} />;
  if (!doc || !pages) return <Loading text="Loading document…" />;

  // Group entities by type: { PER: [...], ORG: [...] }, then list known types first in a fixed order.
  const groups: Record<string, Entity[]> = {};
  for (const e of doc.entities ?? []) (groups[e.type] ??= []).push(e);
  const order = [...Object.keys(ENTITY_TYPES), ...Object.keys(groups).filter((t) => !(t in ENTITY_TYPES))];

  return (
    <div className="space-y-6">
      <header className="space-y-3 rounded border border-border bg-card p-4">
        <div className="flex flex-wrap items-center gap-3">
          <Link href="/documents" className="text-sm text-accent hover:underline">
            ← Documents
          </Link>
          <a href={fileUrl(doc.id)} className="ml-auto rounded border border-border px-3 py-1.5 text-sm" download>
            Download original
          </a>
        </div>
        <h1 className="break-all text-xl font-semibold">{doc.filename}</h1>
        <div className="flex flex-wrap items-center gap-2 text-sm text-muted">
          <StatusBadge status={doc.status} />
          <span>
            {doc.num_pages} pages · {doc.num_chunks} chunks
          </span>
          {doc.label && (
            <span className="rounded bg-accent/15 px-2 py-0.5 text-fg">
              Topic: {doc.label}
              {doc.label_score != null && ` (${(doc.label_score * 100).toFixed(0)}%)`}
            </span>
          )}
          <span>enrichment: {doc.enrich_status}</span>
        </div>

        {doc.summary && (
          <div>
            <h2 className="text-xs font-semibold uppercase text-muted">Summary</h2>
            <p className="mt-1 leading-relaxed">{doc.summary}</p>
          </div>
        )}

        {Object.keys(groups).length > 0 && (
          <div className="space-y-2">
            <h2 className="text-xs font-semibold uppercase text-muted">Entities</h2>
            {order.filter((type) => groups[type]).map((type) => (
              <div key={type} className="flex flex-wrap items-center gap-1 text-sm">
                <span className="mr-1 w-28 shrink-0 text-muted">{ENTITY_TYPES[type] ?? type}</span>
                {groups[type]!.map((e) => (
                  <span key={e.text} className="rounded-full border border-border px-2 py-0.5">
                    {e.text} <span className="text-muted">×{e.count}</span>
                  </span>
                ))}
              </div>
            ))}
          </div>
        )}
      </header>

      {pages.length === 0 && (
        <p className="text-sm text-muted">No pages yet - the document may still be processing.</p>
      )}
      {pages.map((p) => {
        const range = p.number === target.page ? locate(p, target) : null;
        return (
          <section
            key={p.number}
            id={`page-${p.number}`}
            aria-label={`Page ${p.number}`}
            className="scroll-mt-20 rounded border border-border bg-card p-4"
          >
            <h2 className="mb-2 text-xs font-semibold uppercase text-muted">Page {p.number}</h2>
            <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">
              {range ? (
                <>
                  {p.text.slice(0, range[0])}
                  <mark ref={markRef}>{p.text.slice(range[0], range[1])}</mark>
                  {p.text.slice(range[1])}
                </>
              ) : (
                p.text
              )}
            </p>
          </section>
        );
      })}
    </div>
  );
}
