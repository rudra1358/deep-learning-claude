"use client";

import { useEffect, useState } from "react";
import { Bookmark, BookmarkCheck, CheckCircle2, Circle } from "lucide-react";
import { progress, useProgress } from "@/lib/progress";

export function LessonActions({ id, compact = false }: { id: string; compact?: boolean }) {
  const p = useProgress();
  const done = !!p.completed[id];
  const marked = p.bookmarks.includes(id);
  useEffect(() => progress.visit(id), [id]);
  return (
    <div className={`flex ${compact ? "flex-row" : "flex-col"} gap-2`}>
      <button onClick={() => progress.toggleComplete(id)} className={`sim-btn ${done ? "!border-accent !bg-accent-soft" : ""}`} aria-pressed={done}>
        {done ? <CheckCircle2 size={15} className="text-accent" /> : <Circle size={15} />} {done ? "Completed" : "Mark as complete"}
      </button>
      <button onClick={() => progress.toggleBookmark(id)} className="sim-btn" aria-pressed={marked}>
        {marked ? <BookmarkCheck size={15} className="text-accent" /> : <Bookmark size={15} />} {marked ? "Bookmarked" : "Bookmark"}
      </button>
    </div>
  );
}

export function Toc({ items }: { items: { id: string; label: string }[] }) {
  const [active, setActive] = useState(items[0]?.id);
  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (vis[0]) setActive(vis[0].target.id);
      },
      { rootMargin: "-80px 0px -65% 0px" },
    );
    items.forEach((i) => {
      const el = document.getElementById(i.id);
      if (el) obs.observe(el);
    });
    return () => obs.disconnect();
  }, [items]);
  return (
    <nav aria-label="On this page" className="text-sm">
      <div className="mb-2 text-xs font-medium uppercase tracking-wider text-faint">On this page</div>
      <ul className="space-y-0.5 border-l border-border">
        {items.map((i, n) => (
          <li key={i.id}>
            <a href={`#${i.id}`} className={`-ml-px block border-l py-0.5 pl-3 ${active === i.id ? "border-accent text-text" : "border-transparent text-muted hover:text-text"}`}>
              <span className="mr-1.5 font-mono text-[0.65rem] text-faint">{String(n + 1).padStart(2, "0")}</span>
              {i.label}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
