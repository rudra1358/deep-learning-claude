"use client";

import Link from "next/link";
import { MODULES, TOTAL_TOPICS, getTopic } from "@/content/curriculum";
import { progress, streak, useProgress } from "@/lib/progress";

export function ProgressDashboard({ published, projects }: { published: string[]; projects: { id: string; title: string; total: number }[] }) {
  const p = useProgress();
  const done = Object.keys(p.completed);
  const pub = new Set(published);
  const quizzes = Object.entries(p.quizzes);
  const avg = quizzes.length ? quizzes.reduce((s, [, q]) => s + q.score / q.total, 0) / quizzes.length : 0;

  // last 12 weeks activity grid
  const days: string[] = [];
  const d = new Date();
  for (let i = 83; i >= 0; i--) {
    const x = new Date(d);
    x.setDate(d.getDate() - i);
    days.push(x.toISOString().slice(0, 10));
  }
  const active = new Set(p.activeDays);

  return (
    <div className="space-y-10 py-10">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Topics completed" value={`${done.length}`} sub={`of ${TOTAL_TOPICS} (${published.length} ready)`} />
        <Stat label="Learning streak" value={`${streak(p.activeDays)}`} sub="days in a row" />
        <Stat label="Quizzes" value={`${quizzes.length}`} sub={quizzes.length ? `average best score ${(avg * 100).toFixed(0)}%` : "none taken yet"} />
        <Stat label="Bookmarks" value={`${p.bookmarks.length}`} sub="saved topics" />
      </div>

      <section>
        <h2 className="mb-3 font-serif text-xl font-semibold">Activity — last 12 weeks</h2>
        <div className="grid w-fit grid-flow-col grid-rows-7 gap-1">
          {days.map((day) => (
            <div key={day} title={day} className={`h-3.5 w-3.5 rounded-sm ${active.has(day) ? "bg-accent" : "bg-surface-2"}`} />
          ))}
        </div>
      </section>

      <section>
        <h2 className="mb-3 font-serif text-xl font-semibold">Modules</h2>
        <div className="grid gap-2 md:grid-cols-2">
          {MODULES.map((m) => {
            const total = m.topics.length;
            const c = m.topics.filter((t) => p.completed[`${m.id}/${t.slug}`]).length;
            const avail = m.topics.filter((t) => pub.has(`${m.id}/${t.slug}`)).length;
            return (
              <Link key={m.id} href={`/curriculum#${m.id}`} className="rounded-lg border border-border bg-surface p-3 hover:border-border-strong">
                <div className="flex justify-between text-sm">
                  <span>
                    <span className="mr-2 font-mono text-xs text-faint">{String(m.number).padStart(2, "0")}</span>
                    {m.title}
                  </span>
                  <span className="font-mono text-xs text-faint">
                    {c}/{total}
                  </span>
                </div>
                <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-surface-2">
                  <div className="h-full rounded-full bg-accent" style={{ width: `${(c / total) * 100}%` }} />
                </div>
                {avail > 0 && <div className="mt-1 text-[0.68rem] text-faint">{avail} full lesson{avail > 1 ? "s" : ""} ready</div>}
              </Link>
            );
          })}
        </div>
      </section>

      <div className="grid gap-8 lg:grid-cols-2">
        <section>
          <h2 className="mb-3 font-serif text-xl font-semibold">Quiz scores</h2>
          {quizzes.length === 0 ? (
            <p className="text-sm text-faint">No scores yet. Try the Knowledge Check at the end of any lesson.</p>
          ) : (
            <ul className="divide-y divide-border text-sm">
              {quizzes.map(([k, q]) => {
                const tid = k.split(":")[0];
                return (
                  <li key={k} className="flex justify-between py-2">
                    <Link href={`/learn/${tid}#practice`} className="hover:text-accent">
                      {getTopic(tid)?.topic.title ?? tid}
                    </Link>
                    <span className="font-mono">
                      {q.score}/{q.total}
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </section>
        <section>
          <h2 className="mb-3 font-serif text-xl font-semibold">Bookmarks</h2>
          {p.bookmarks.length === 0 ? (
            <p className="text-sm text-faint">Nothing saved yet. Use the Bookmark button on any lesson.</p>
          ) : (
            <ul className="divide-y divide-border text-sm">
              {p.bookmarks.map((b) => (
                <li key={b} className="py-2">
                  <Link href={`/learn/${b}`} className="hover:text-accent">
                    {getTopic(b)?.topic.title ?? b}
                  </Link>
                </li>
              ))}
            </ul>
          )}
          <h2 className="mt-8 mb-3 font-serif text-xl font-semibold">Projects</h2>
          <ul className="space-y-2 text-sm">
            {projects.map((pr) => {
              const c = (p.projects[pr.id] ?? []).length;
              return (
                <li key={pr.id}>
                  <Link href={`/projects#${pr.id}`} className="flex justify-between hover:text-accent">
                    <span>{pr.title}</span>
                    <span className="font-mono text-xs text-faint">
                      {c}/{pr.total} milestones
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </section>
      </div>

      <section className="border-t border-border pt-6 text-sm text-muted">
        Your progress is saved in this browser{process.env.NEXT_PUBLIC_API_URL ? " and synced to the server" : ""}.{" "}
        <button
          className="text-clay underline underline-offset-2"
          onClick={() => {
            if (confirm("Reset all your progress? You can't undo this.")) progress.reset();
          }}
        >
          Reset progress
        </button>
      </section>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="card p-5">
      <div className="text-xs uppercase tracking-wider text-faint">{label}</div>
      <div className="mt-1 font-serif text-3xl font-semibold">{value}</div>
      <div className="text-xs text-muted">{sub}</div>
    </div>
  );
}
