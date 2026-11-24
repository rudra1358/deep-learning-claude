"use client";

import Link from "next/link";
import { useProgress, streak } from "@/lib/progress";
import { getTopic, TOTAL_TOPICS } from "@/content/curriculum";

export function ProgressPreview({ firstLesson }: { firstLesson: string }) {
  const p = useProgress();
  const done = Object.keys(p.completed).length;
  const last = p.lastVisited ? getTopic(p.lastVisited) : undefined;
  const pct = Math.round((done / TOTAL_TOPICS) * 100);
  return (
    <div className="card p-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="text-xs font-medium uppercase tracking-wider text-faint">Your progress</div>
          <div className="mt-1 font-serif text-2xl font-semibold">
            {done} <span className="text-base font-normal text-muted">of {TOTAL_TOPICS} topics</span>
          </div>
        </div>
        <div className="flex gap-6 text-sm">
          <div>
            <div className="font-mono text-lg">{streak(p.activeDays)}</div>
            <div className="text-xs text-faint">day streak</div>
          </div>
          <div>
            <div className="font-mono text-lg">{Object.keys(p.quizzes).length}</div>
            <div className="text-xs text-faint">quizzes taken</div>
          </div>
          <div>
            <div className="font-mono text-lg">{p.bookmarks.length}</div>
            <div className="text-xs text-faint">bookmarks</div>
          </div>
        </div>
      </div>
      <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-surface-2" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className="h-full rounded-full bg-accent" style={{ width: `${Math.max(pct, done ? 2 : 0)}%` }} />
      </div>
      <div className="mt-4 flex flex-wrap gap-3 text-sm">
        {last ? (
          <Link href={`/learn/${last.module.id}/${last.topic.slug}`} className="sim-btn primary">
            Continue: {last.topic.title}
          </Link>
        ) : (
          <Link href={`/learn/${firstLesson}`} className="sim-btn primary">
            Begin the first lesson
          </Link>
        )}
        <Link href="/progress" className="sim-btn">
          Open dashboard
        </Link>
      </div>
      <p className="mt-3 text-xs text-faint">Progress is saved in this browser — no account needed.</p>
    </div>
  );
}
