"use client";

import Link from "next/link";
import { CheckCircle2, Circle } from "lucide-react";
import { useProgress } from "@/lib/progress";

export function TopicRow({ id, title, summary, subtopics, level, full }: { id: string; title: string; summary: string; subtopics: string[]; level: string; full: boolean }) {
  const p = useProgress();
  const done = !!p.completed[id];
  return (
    <li className="py-3">
      <Link href={`/learn/${id}`} className="group flex gap-3">
        {done ? <CheckCircle2 size={17} className="mt-1 shrink-0 text-accent" /> : <Circle size={17} className="mt-1 shrink-0 text-border-strong" />}
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium group-hover:text-accent">{title}</span>
            {full && <span className="rounded bg-accent-soft px-1.5 py-0.5 text-[0.68rem] text-accent-strong">full lesson</span>}
            <span className="text-[0.68rem] uppercase tracking-wider text-faint">{level}</span>
          </div>
          <p className="mt-0.5 text-sm text-muted">{summary}</p>
          <p className="mt-1 line-clamp-1 text-xs text-faint">{subtopics.join(" · ")}</p>
        </div>
      </Link>
    </li>
  );
}
