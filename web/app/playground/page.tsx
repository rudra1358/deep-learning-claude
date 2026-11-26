import type { Metadata } from "next";
import Link from "next/link";
import { SIMS } from "@/components/sims/registry";
import { getTopic } from "@/content/curriculum";
import { Container, PageHeader } from "@/components/pages/PageHeader";

export const metadata: Metadata = { title: "Interactive Playground" };

export default function PlaygroundPage() {
  const entries = Object.entries(SIMS).sort((a, b) => Number(!!b[1].flagship) - Number(!!a[1].flagship));
  return (
    <Container>
      <PageHeader eyebrow="Playground" title="Every simulator in one place">
        Each simulator runs the real algorithm — the numbers you see are really computed. Every one has play, pause, step, reset and speed controls.
      </PageHeader>
      <div className="grid gap-4 py-10 sm:grid-cols-2 lg:grid-cols-3">
        {entries.map(([id, s]) => {
          const lesson = getTopic(s.lesson);
          return (
            <Link key={id} href={`/playground/${id}`} className="card group flex flex-col p-5 hover:border-border-strong">
              {s.flagship && <div className="mb-1 text-[0.68rem] font-medium uppercase tracking-wider text-accent">Flagship</div>}
              <div className="font-serif text-lg font-semibold group-hover:text-accent">{s.title}</div>
              <p className="mt-1 flex-1 text-sm text-muted">{s.blurb}</p>
              {lesson && <div className="mt-3 text-xs text-faint">From: {lesson.topic.title}</div>}
            </Link>
          );
        })}
      </div>
    </Container>
  );
}
