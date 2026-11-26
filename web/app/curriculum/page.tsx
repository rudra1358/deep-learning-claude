import type { Metadata } from "next";
import { MODULES, PHASES } from "@/content/curriculum";
import { hasLesson } from "@/content/lessons";
import { Container, PageHeader } from "@/components/pages/PageHeader";
import { TopicRow } from "@/components/pages/TopicRow";

export const metadata: Metadata = { title: "Curriculum" };

export default function CurriculumPage() {
  return (
    <Container>
      <PageHeader eyebrow="All lessons" title="21 modules, in the order to learn them">
        Each module only uses what came before it. Topics marked <span className="rounded bg-accent-soft px-1.5 text-sm text-accent-strong">full lesson</span> are ready, with all 16 sections. The rest show what they will cover while they are being written.
      </PageHeader>
      {PHASES.map((ph) => (
        <section key={ph} className="py-10">
          <h2 className="mb-6 text-xs font-medium uppercase tracking-[0.15em] text-faint">{ph}</h2>
          <div className="space-y-6">
            {MODULES.filter((m) => m.phase === ph).map((m) => (
              <div key={m.id} id={m.id} className="card scroll-mt-20 p-6">
                <div className="flex flex-wrap items-baseline gap-x-3">
                  <span className="font-mono text-sm text-faint">Module {String(m.number).padStart(2, "0")}</span>
                  <h3 className="font-serif text-2xl font-semibold">{m.title}</h3>
                </div>
                <p className="mt-1 text-muted">{m.tagline}</p>
                <ul className="mt-4 divide-y divide-border">
                  {m.topics.map((t) => (
                    <TopicRow key={t.slug} id={`${m.id}/${t.slug}`} title={t.title} summary={t.summary} subtopics={t.subtopics} level={t.level} full={hasLesson(`${m.id}/${t.slug}`)} />
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>
      ))}
    </Container>
  );
}
