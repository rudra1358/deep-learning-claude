"use client";

import { useState, type ReactNode } from "react";
import { ChevronRight, Lightbulb } from "lucide-react";

export function Details({ summary, children }: { summary: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="my-4 rounded-lg border border-border">
      <button onClick={() => setOpen((o) => !o)} aria-expanded={open} className="flex w-full items-center gap-2 px-4 py-2.5 text-left text-sm font-medium hover:bg-surface-2">
        <ChevronRight size={16} className={`transition-transform ${open ? "rotate-90" : ""}`} />
        {summary}
      </button>
      {open && <div className="border-t border-border px-4 pb-2">{children}</div>}
    </div>
  );
}

export function Tabs({ labels, children }: { labels: string[]; children: ReactNode[] }) {
  const [i, setI] = useState(0);
  return (
    <div className="my-5">
      <div role="tablist" className="flex gap-1 overflow-x-auto border-b border-border">
        {/* scrolls horizontally when labels exceed the width (e.g. 4+ framework tabs on a phone) */}
        {labels.map((l, k) => (
          <button
            key={l}
            role="tab"
            aria-selected={i === k}
            onClick={() => setI(k)}
            className={`-mb-px shrink-0 whitespace-nowrap border-b-2 px-3 py-2 text-sm ${i === k ? "border-accent font-medium text-text" : "border-transparent text-muted hover:text-text"}`}
          >
            {l}
          </button>
        ))}
      </div>
      <div>{children[i]}</div>
    </div>
  );
}

export function Exercise({ title, prompt, hint, children }: { title: string; prompt: ReactNode; hint?: ReactNode; children: ReactNode }) {
  const [showHint, setShowHint] = useState(false);
  const [showSol, setShowSol] = useState(false);
  return (
    <div className="card my-5 p-5">
      <div className="mb-1 text-xs font-medium uppercase tracking-wider text-gold">Exercise</div>
      <div className="mb-2 font-medium">{title}</div>
      <div className="text-[0.95rem]">{prompt}</div>
      <div className="mt-3 flex gap-2">
        {hint && (
          <button className="sim-btn" onClick={() => setShowHint((s) => !s)}>
            <Lightbulb size={14} /> {showHint ? "Hide hint" : "Hint"}
          </button>
        )}
        <button className="sim-btn" onClick={() => setShowSol((s) => !s)}>
          {showSol ? "Hide solution" : "Show solution"}
        </button>
      </div>
      {showHint && <div className="mt-3 rounded-lg bg-gold-soft p-3 text-sm">{hint}</div>}
      {showSol && <div className="mt-3 border-t border-border pt-2">{children}</div>}
    </div>
  );
}
