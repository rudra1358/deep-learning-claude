import type { Metadata } from "next";
import Link from "next/link";
import { GLOSSARY } from "@/content/glossary";
import { Container, PageHeader } from "@/components/pages/PageHeader";
import { Inline, MathBlock } from "@/components/content/Inline";

export const metadata: Metadata = { title: "Glossary" };

export default function GlossaryPage() {
  const sorted = [...GLOSSARY].sort((a, b) => a.term.localeCompare(b.term));
  const letters = Array.from(new Set(sorted.map((g) => g.term[0].toUpperCase())));
  return (
    <Container>
      <PageHeader eyebrow="Glossary" title="Every term, defined once">
        Concise definitions with the defining formula and a link to the lesson that teaches the concept properly.
      </PageHeader>
      <nav className="sticky top-14 z-10 -mx-4 flex flex-wrap gap-1 border-b border-border bg-bg/95 px-4 py-2 backdrop-blur" aria-label="Letters">
        {letters.map((l) => (
          <a key={l} href={`#letter-${l}`} className="rounded px-2 py-0.5 font-mono text-sm text-muted hover:bg-surface-2 hover:text-text">
            {l}
          </a>
        ))}
      </nav>
      {letters.map((l) => (
        <section key={l} id={`letter-${l}`} className="scroll-mt-28 py-6">
          <h2 className="mb-2 font-serif text-2xl text-faint">{l}</h2>
          <dl className="divide-y divide-border">
            {sorted
              .filter((g) => g.term[0].toUpperCase() === l)
              .map((g) => (
                <div key={g.key} id={g.key} className="grid scroll-mt-28 grid-cols-1 gap-2 py-4 md:grid-cols-[220px_minmax(0,1fr)]">
                  <dt className="font-medium">{g.term}</dt>
                  <dd className="text-sm text-muted">
                    <Inline text={g.definition} />
                    {g.tex && <MathBlock tex={g.tex} />}
                    {g.lesson && (
                      <Link href={`/learn/${g.lesson}`} className="text-xs text-accent">
                        Learn more →
                      </Link>
                    )}
                  </dd>
                </div>
              ))}
          </dl>
        </section>
      ))}
    </Container>
  );
}
