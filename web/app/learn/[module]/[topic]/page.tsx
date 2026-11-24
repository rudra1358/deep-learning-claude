import Link from "next/link";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { ArrowLeft, ArrowRight, Clock, PenLine } from "lucide-react";
import { LEARNING_ORDER, MODULES, getTopic } from "@/content/curriculum";
import { LESSONS, getLesson } from "@/content/lessons";
import { SIMS } from "@/components/sims/registry";
import { SECTION_ORDER } from "@/lib/types";
import { Blocks } from "@/components/content/Blocks";
import { CurriculumSidebar } from "@/components/lesson/CurriculumSidebar";
import { LessonActions, Toc } from "@/components/lesson/LessonAside";
import { SimEmbed } from "@/components/sims/SimEmbed";

type Params = { module: string; topic: string };

export function generateStaticParams(): Params[] {
  return MODULES.flatMap((m) => m.topics.map((t) => ({ module: m.id, topic: t.slug })));
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { module, topic } = await params;
  const found = getTopic(`${module}/${topic}`);
  return found ? { title: found.topic.title, description: found.topic.summary } : {};
}

const LEVEL_LABEL = { beginner: "Beginner", intermediate: "Intermediate", advanced: "Advanced", research: "Research" };

export default async function LessonPage({ params }: { params: Promise<Params> }) {
  const { module, topic } = await params;
  const id = `${module}/${topic}`;
  const found = getTopic(id);
  if (!found) notFound();
  const { module: mod, topic: tp } = found;
  const lesson = getLesson(id);
  const idx = LEARNING_ORDER.indexOf(id);
  const prev = idx > 0 ? getTopic(LEARNING_ORDER[idx - 1]) : undefined;
  const next = idx < LEARNING_ORDER.length - 1 ? getTopic(LEARNING_ORDER[idx + 1]) : undefined;
  const available = LESSONS.map((l) => l.id);
  const toc = SECTION_ORDER.map(([key, label]) => ({ id: key, label }));
  const relatedSims = Object.entries(SIMS).filter(([, s]) => s.lesson === id);

  return (
    <div className="mx-auto grid max-w-[1400px] grid-cols-1 gap-8 px-4 lg:grid-cols-[260px_minmax(0,1fr)] xl:grid-cols-[260px_minmax(0,1fr)_220px]">
      <aside className="hidden lg:block">
        <div className="sticky top-14 max-h-[calc(100vh-3.5rem)] overflow-y-auto py-6 pr-2">
          <CurriculumSidebar current={id} available={available} />
        </div>
      </aside>

      <article className="min-w-0 py-8">
        <nav aria-label="Breadcrumb" className="mb-4 flex flex-wrap items-center gap-1.5 text-xs text-faint">
          <Link href="/curriculum" className="hover:text-text">
            Curriculum
          </Link>
          <span>/</span>
          <Link href={`/curriculum#${mod.id}`} className="hover:text-text">
            Module {mod.number}: {mod.title}
          </Link>
          <span>/</span>
          <span className="text-muted">{tp.title}</span>
        </nav>

        <header className="mb-8 border-b border-border pb-6">
          <h1 className="font-serif text-3xl leading-tight font-semibold md:text-[2.4rem]">{lesson?.title ?? tp.title}</h1>
          <p className="mt-3 max-w-3xl text-lg text-muted">{lesson?.summary ?? tp.summary}</p>
          <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-muted">
            <span className="rounded-full border border-border px-2.5 py-1">{LEVEL_LABEL[lesson?.level ?? tp.level]}</span>
            {lesson && (
              <span className="flex items-center gap-1">
                <Clock size={13} /> ~{lesson.minutes} min
              </span>
            )}
            <span>Module {mod.number} · {mod.phase}</span>
          </div>
          {lesson && (
            <div className="mt-5 xl:hidden">
              <LessonActions id={id} compact />
            </div>
          )}
        </header>

        {lesson ? (
          <div className="prose-lesson max-w-[46rem]">
            {SECTION_ORDER.map(([key, label], n) => (
              <section key={key} id={key} className="scroll-mt-20 border-b border-border/70 pb-6 mb-8 last:border-0">
                <h2 className="mb-3 flex items-baseline gap-3 font-serif text-2xl font-semibold">
                  <span className="font-mono text-sm text-faint">{String(n + 1).padStart(2, "0")}</span>
                  {label}
                </h2>
                <Blocks blocks={lesson.sections[key]} ctx={`${id}:${key}`} />
              </section>
            ))}
          </div>
        ) : (
          <div className="max-w-[46rem]">
            <div className="card mb-6 flex gap-3 p-5">
              <PenLine size={20} className="mt-0.5 shrink-0 text-gold" />
              <div className="text-sm">
                <div className="font-medium">This lesson is being written.</div>
                <p className="mt-1 text-muted">
                  The full 16-part lesson (intuition, mechanism, derivations, simulator, NumPy and PyTorch code, exercises and interview questions) is not published yet. Below is its syllabus so you can see exactly what it will cover and study it from the references in the meantime.
                </p>
              </div>
            </div>
            <h2 className="mb-3 font-serif text-xl font-semibold">Syllabus</h2>
            <ul className="grid gap-2 sm:grid-cols-2">
              {tp.subtopics.map((s) => (
                <li key={s} className="rounded-lg border border-border bg-surface px-3 py-2 text-sm capitalize">
                  {s}
                </li>
              ))}
            </ul>
            {relatedSims.map(([sid]) => (
              <SimEmbed key={sid} id={sid as keyof typeof SIMS} />
            ))}
          </div>
        )}

        <nav className="mt-12 grid gap-3 sm:grid-cols-2" aria-label="Previous and next lesson">
          {prev ? (
            <Link href={`/learn/${prev.module.id}/${prev.topic.slug}`} className="card group p-4 hover:border-border-strong">
              <div className="flex items-center gap-1 text-xs text-faint">
                <ArrowLeft size={13} /> Previous
              </div>
              <div className="mt-1 font-medium group-hover:text-accent">{prev.topic.title}</div>
            </Link>
          ) : (
            <span />
          )}
          {next && (
            <Link href={`/learn/${next.module.id}/${next.topic.slug}`} className="card group p-4 text-right hover:border-border-strong">
              <div className="flex items-center justify-end gap-1 text-xs text-faint">
                Next <ArrowRight size={13} />
              </div>
              <div className="mt-1 font-medium group-hover:text-accent">{next.topic.title}</div>
            </Link>
          )}
        </nav>
      </article>

      {lesson && (
        <aside className="hidden xl:block">
          <div className="sticky top-14 space-y-6 py-8">
            <LessonActions id={id} />
            <Toc items={toc} />
          </div>
        </aside>
      )}
    </div>
  );
}
