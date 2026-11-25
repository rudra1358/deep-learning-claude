"use client";

import Link from "next/link";
import { useState } from "react";
import type { Source } from "@/lib/api";
import { pageLabel, viewerHref } from "@/lib/text";

// Renders text with [start, end) wrapped in <mark>. Used for the answer span and the viewer.
export function Highlighted({ text, start, end }: { text: string; start: number; end: number }) {
  return (
    <>
      {text.slice(0, start)}
      <mark>{text.slice(start, end)}</mark>
      {text.slice(end)}
    </>
  );
}

const PREVIEW = 300; // characters shown before "Show more"

interface Props {
  source: Source;
  span?: { start: number; end: number } | null; // answer span inside source.text, if this is the cited source
  why?: string; // optional extra line (the Search page shows retriever ranks here)
  id?: string; // DOM id, so [n] chips in an answer can scroll to this card
}

// One numbered retrieved chunk: filename, page, score, a snippet and a link into the viewer.
export default function SourceCard({ source, span, why, id }: Props) {
  const [expanded, setExpanded] = useState(false);
  const text = source.text;

  // Which part of the chunk to show. With a highlight, centre the window on the answer;
  // otherwise show the beginning. "Show more" reveals the whole chunk.
  let from = 0;
  let to = Math.min(text.length, PREVIEW);
  if (span) {
    from = Math.max(0, span.start - 150);
    to = Math.min(text.length, span.end + 150);
  }
  if (expanded) {
    from = 0;
    to = text.length;
  }
  const shown = text.slice(from, to);
  const truncated = from > 0 || to < text.length;

  return (
    <article id={id} className="scroll-mt-20 rounded border border-border bg-card p-3 text-sm">
      <header className="mb-1 flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <span className="rounded bg-accent px-1.5 text-xs font-semibold text-accent-fg">{source.n}</span>
        <Link href={viewerHref(source)} className="min-w-0 break-all font-medium text-accent hover:underline">
          {source.filename}
        </Link>
        <span className="text-muted">{pageLabel(source)}</span>
        <span className="ml-auto text-xs text-muted" title="Relevance score (higher = more relevant)">
          score {source.score.toFixed(3)}
        </span>
      </header>
      {why && <p className="mb-1 text-xs text-muted">{why}</p>}
      <p className="whitespace-pre-wrap break-words leading-relaxed">
        {from > 0 && "…"}
        {/* offsets are relative to the whole chunk, so shift them by `from` for the shown window */}
        {span ? <Highlighted text={shown} start={span.start - from} end={span.end - from} /> : shown}
        {to < text.length && "…"}
      </p>
      {(truncated || expanded) && (
        <button type="button" onClick={() => setExpanded(!expanded)} className="mt-1 text-xs text-accent hover:underline">
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </article>
  );
}
