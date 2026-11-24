"use client";

import Link from "next/link";
import { useState } from "react";
import { CheckCircle2, ChevronRight, Circle } from "lucide-react";
import { MODULES } from "@/content/curriculum";
import { useProgress } from "@/lib/progress";

export function CurriculumSidebar({ current, available }: { current: string; available: string[] }) {
  const p = useProgress();
  const curMod = current.split("/")[0];
  const [open, setOpen] = useState<Record<string, boolean>>({ [curMod]: true });
  const avail = new Set(available);

  return (
    <nav aria-label="Curriculum" className="text-sm">
      {MODULES.map((m) => {
        const isOpen = open[m.id] ?? false;
        const done = m.topics.filter((t) => p.completed[`${m.id}/${t.slug}`]).length;
        return (
          <div key={m.id} className="mb-0.5">
            <button onClick={() => setOpen((o) => ({ ...o, [m.id]: !isOpen }))} className="flex w-full items-start gap-1.5 rounded-md px-2 py-1.5 text-left hover:bg-surface-2" aria-expanded={isOpen}>
              <ChevronRight size={14} className={`mt-0.5 shrink-0 text-faint transition-transform ${isOpen ? "rotate-90" : ""}`} />
              <span className={`flex-1 ${m.id === curMod ? "font-medium text-text" : "text-muted"}`}>
                <span className="mr-1 font-mono text-xs text-faint">{String(m.number).padStart(2, "0")}</span>
                {m.title}
              </span>
              {done > 0 && (
                <span className="mt-0.5 font-mono text-[0.65rem] text-faint">
                  {done}/{m.topics.length}
                </span>
              )}
            </button>
            {isOpen && (
              <ul className="mb-2 ml-5 border-l border-border">
                {m.topics.map((t) => {
                  const id = `${m.id}/${t.slug}`;
                  const isCur = id === current;
                  return (
                    <li key={id}>
                      <Link href={`/learn/${id}`} className={`-ml-px flex items-start gap-2 border-l py-1 pl-3 pr-1 ${isCur ? "border-accent text-text" : "border-transparent text-muted hover:text-text"}`} aria-current={isCur ? "page" : undefined}>
                        {p.completed[id] ? <CheckCircle2 size={13} className="mt-1 shrink-0 text-accent" /> : <Circle size={13} className="mt-1 shrink-0 text-border-strong" />}
                        <span className={avail.has(id) ? "" : "opacity-60"}>{t.title}</span>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        );
      })}
    </nav>
  );
}
