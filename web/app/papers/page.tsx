import type { Metadata } from "next";
import Link from "next/link";
import { ExternalLink } from "lucide-react";
import { PAPERS } from "@/content/papers";
import { getTopic } from "@/content/curriculum";
import { Container, PageHeader } from "@/components/pages/PageHeader";

export const metadata: Metadata = { title: "Research Papers" };

export default function PapersPage() {
  const eras = Array.from(new Set(PAPERS.map((p) => p.era)));
  return (
    <Container>
      <PageHeader eyebrow="Research papers" title="The papers that shaped deep learning">
        Each entry sums up what the paper added in one sentence, and links to the lessons that teach the idea. Grouped by theme, then sorted by year.
      </PageHeader>
      {eras.map((era) => (
        <section key={era} className="py-8">
          <h2 className="mb-4 font-serif text-2xl font-semibold">{era}</h2>
          <ul className="divide-y divide-border">
            {PAPERS.filter((p) => p.era === era)
              .sort((a, b) => a.year - b.year)
              .map((p) => (
                <li key={p.key} id={p.key} className="scroll-mt-20 py-4">
                  <div className="flex flex-wrap items-baseline gap-x-3">
                    <span className="font-mono text-sm text-faint">{p.year}</span>
                    {p.url ? (
                      <a href={p.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 font-medium hover:text-accent">
                        {p.title} <ExternalLink size={12} className="text-faint" />
                      </a>
                    ) : (
                      <span className="font-medium">{p.title}</span>
                    )}
                  </div>
                  <div className="text-sm text-faint">
                    {p.authors}
                    {p.venue ? ` · ${p.venue}` : ""}
                  </div>
                  <p className="mt-1 text-sm text-muted">{p.contribution}</p>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {p.topics.map((t) => (
                      <Link key={t} href={`/learn/${t}`} className="rounded-full border border-border px-2 py-0.5 text-xs text-muted hover:border-accent hover:text-text">
                        {getTopic(t)?.topic.title ?? t}
                      </Link>
                    ))}
                  </div>
                </li>
              ))}
          </ul>
        </section>
      ))}
    </Container>
  );
}
