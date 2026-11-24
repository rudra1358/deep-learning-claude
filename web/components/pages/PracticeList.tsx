"use client";

import Link from "next/link";
import { useProgress } from "@/lib/progress";

interface Item {
  id: string;
  title: string;
  module: string;
  questions: number;
  exercises: number;
  interview: number;
  quizKey: string;
}

export function PracticeList({ items }: { items: Item[] }) {
  const p = useProgress();
  return (
    <div className="grid gap-3 py-10 md:grid-cols-2">
      {items.map((it) => {
        const best = Object.entries(p.quizzes).find(([k]) => k.startsWith(it.quizKey))?.[1];
        return (
          <Link key={it.id} href={`/learn/${it.id}#practice`} className="card p-5 hover:border-border-strong">
            <div className="text-xs text-faint">{it.module}</div>
            <div className="mt-1 font-medium">{it.title}</div>
            <div className="mt-3 flex flex-wrap gap-4 text-xs text-muted">
              <span>{it.questions} quiz questions</span>
              <span>{it.exercises} exercises</span>
              <span>{it.interview} interview questions</span>
            </div>
            <div className="mt-3 text-sm">
              {best ? (
                <span className="text-accent">
                  Best score: {best.score}/{best.total}
                </span>
              ) : (
                <span className="text-faint">Not attempted</span>
              )}
            </div>
          </Link>
        );
      })}
    </div>
  );
}
