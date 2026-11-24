import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { LEARNING_ORDER, MODULES, PHASES } from "@/content/curriculum";
import { LESSONS, hasLesson } from "@/content/lessons";
import { SIMS } from "@/components/sims/registry";
import { PROJECTS } from "@/content/projects";
import { SECTION_ORDER } from "@/lib/types";
import { HeroNetwork } from "@/components/home/HeroNetwork";
import { ProgressPreview } from "@/components/home/ProgressPreview";

export default function Home() {
  const firstLesson = LEARNING_ORDER.find(hasLesson) ?? LEARNING_ORDER[0];
  const flagship = Object.entries(SIMS).filter(([, s]) => s.flagship);
  const more = Object.entries(SIMS).filter(([, s]) => !s.flagship).slice(0, 6);
  const topicCount = MODULES.reduce((n, m) => n + m.topics.length, 0);

  return (
    <div className="mx-auto max-w-[1200px] px-4">
      {/* Hero */}
      <section className="grid items-center gap-10 py-16 md:py-24 lg:grid-cols-[1.1fr_1fr]">
        <div>
          <p className="mb-4 text-sm text-accent">An interactive deep learning curriculum</p>
          <h1 className="font-serif text-4xl leading-[1.1] font-semibold tracking-tight md:text-[3.4rem]">
            Learn deep learning
            <br />
            by watching it compute.
          </h1>
          <p className="mt-6 max-w-xl text-lg leading-relaxed text-muted">
            From vectors and derivatives to backpropagation, CNNs, LSTMs and Transformers. Every concept is explained from first principles, derived mathematically, implemented in Python, and shown in a simulator that runs the real calculation.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href={`/learn/${firstLesson}`} className="inline-flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-sm font-medium text-bg hover:bg-accent-strong">
              Start learning <ArrowRight size={16} />
            </Link>
            <Link href="/curriculum" className="inline-flex items-center gap-2 rounded-lg border border-border-strong bg-surface px-5 py-2.5 text-sm font-medium hover:bg-surface-2">
              Explore the curriculum
            </Link>
          </div>
          <dl className="mt-10 flex gap-8 text-sm">
            <div>
              <dt className="text-faint">Modules</dt>
              <dd className="font-serif text-2xl">{MODULES.length}</dd>
            </div>
            <div>
              <dt className="text-faint">Topics</dt>
              <dd className="font-serif text-2xl">{topicCount}</dd>
            </div>
            <div>
              <dt className="text-faint">Simulators</dt>
              <dd className="font-serif text-2xl">{Object.keys(SIMS).length}</dd>
            </div>
            <div>
              <dt className="text-faint">Full lessons</dt>
              <dd className="font-serif text-2xl">{LESSONS.length}</dd>
            </div>
          </dl>
        </div>
        <figure className="card p-5">
          <HeroNetwork />
          <figcaption className="mt-2 text-center text-xs text-faint">A real forward and backward pass: activations on neurons, gradients δ beneath them.</figcaption>
        </figure>
      </section>

      {/* Roadmap preview */}
      <section className="border-t border-border py-16">
        <SectionHead title="One path, from first principles to research" link={{ href: "/roadmap", label: "Full roadmap" }} />
        <ol className="grid gap-px overflow-hidden rounded-xl border border-border bg-border sm:grid-cols-2 lg:grid-cols-4">
          {PHASES.map((ph, i) => {
            const mods = MODULES.filter((m) => m.phase === ph);
            return (
              <li key={ph} className="bg-surface p-5">
                <div className="font-mono text-xs text-faint">{String(i + 1).padStart(2, "0")}</div>
                <div className="mt-1 font-serif text-lg font-semibold">{ph}</div>
                <ul className="mt-2 space-y-1 text-sm text-muted">
                  {mods.map((m) => (
                    <li key={m.id}>
                      <Link href={`/curriculum#${m.id}`} className="hover:text-text">
                        {m.title}
                      </Link>
                    </li>
                  ))}
                </ul>
              </li>
            );
          })}
          <li className="bg-surface p-5">
            <div className="font-mono text-xs text-faint">08</div>
            <div className="mt-1 font-serif text-lg font-semibold">Industry Projects</div>
            <ul className="mt-2 space-y-1 text-sm text-muted">
              {PROJECTS.map((p) => (
                <li key={p.id}>
                  <Link href={`/projects#${p.id}`} className="hover:text-text">
                    {p.short}
                  </Link>
                </li>
              ))}
            </ul>
          </li>
        </ol>
      </section>

      {/* Flagship simulators */}
      <section className="border-t border-border py-16">
        <SectionHead title="Simulations that run the real algorithm" link={{ href: "/playground", label: "All simulators" }} />
        <div className="grid gap-4 md:grid-cols-3">
          {flagship.map(([id, s]) => (
            <Link key={id} href={`/playground/${id}`} className="card group p-6 hover:border-border-strong">
              <div className="text-xs font-medium uppercase tracking-wider text-accent">Flagship</div>
              <div className="mt-2 font-serif text-xl font-semibold group-hover:text-accent">{s.title}</div>
              <p className="mt-2 text-sm text-muted">{s.blurb}</p>
            </Link>
          ))}
        </div>
        <div className="mt-4 grid gap-x-8 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
          {more.map(([id, s]) => (
            <Link key={id} href={`/playground/${id}`} className="group flex items-baseline justify-between border-b border-border py-2 text-sm">
              <span className="group-hover:text-accent">{s.title}</span>
              <ArrowRight size={14} className="text-faint" />
            </Link>
          ))}
        </div>
      </section>

      {/* Lesson structure */}
      <section className="grid gap-10 border-t border-border py-16 lg:grid-cols-[1fr_1.3fr]">
        <div>
          <h2 className="font-serif text-3xl font-semibold">Every lesson, the same sixteen steps</h2>
          <p className="mt-4 text-muted">
            Concepts are never introduced before their prerequisites. Each lesson moves from intuition to mechanism, to mathematics with worked numbers, to a simulation, to code you can run — first from scratch in NumPy, then in PyTorch and Keras — and ends with practice, interview questions and a recap.
          </p>
          <p className="mt-4 text-muted">All code examples are executed during authoring and show their real output. NumPy examples run right in your browser.</p>
        </div>
        <ol className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
          {SECTION_ORDER.map(([k, label], i) => (
            <li key={k} className="flex gap-3 border-b border-border py-2">
              <span className="font-mono text-xs text-faint">{String(i + 1).padStart(2, "0")}</span>
              {label}
            </li>
          ))}
        </ol>
      </section>

      {/* Projects */}
      <section className="border-t border-border py-16">
        <SectionHead title="Three end-to-end industry projects" link={{ href: "/projects", label: "Project details" }} />
        <div className="grid gap-4 md:grid-cols-3">
          {PROJECTS.map((p) => (
            <Link key={p.id} href={`/projects#${p.id}`} className="card group p-6 hover:border-border-strong">
              <div className="text-xs text-faint">{p.domain}</div>
              <div className="mt-1 font-serif text-lg font-semibold group-hover:text-accent">{p.title}</div>
              <p className="mt-2 text-sm text-muted">{p.summary}</p>
            </Link>
          ))}
        </div>
      </section>

      {/* Progress */}
      <section className="border-t border-border py-16">
        <ProgressPreview firstLesson={firstLesson} />
      </section>
    </div>
  );
}

function SectionHead({ title, link }: { title: string; link: { href: string; label: string } }) {
  return (
    <div className="mb-8 flex flex-wrap items-end justify-between gap-3">
      <h2 className="font-serif text-3xl font-semibold">{title}</h2>
      <Link href={link.href} className="flex items-center gap-1 text-sm text-accent hover:text-accent-strong">
        {link.label} <ArrowRight size={14} />
      </Link>
    </div>
  );
}
