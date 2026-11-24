import type { Metadata } from "next";
import Link from "next/link";
import { MODULES, PHASES } from "@/content/curriculum";
import { hasLesson } from "@/content/lessons";
import { PROJECTS } from "@/content/projects";
import { Container, PageHeader } from "@/components/pages/PageHeader";

export const metadata: Metadata = { title: "Learning Roadmap" };

const PHASE_GOAL: Record<string, string> = {
  Foundations: "Read and write the mathematics of deep learning fluently, and compute it in NumPy.",
  "Neural Networks": "Derive and implement a complete neural network — forward pass, loss, backpropagation, optimizer — from scratch.",
  "Core Architectures": "Understand CNNs, RNNs/LSTMs and attention-based Transformers well enough to implement each one.",
  "Applied Deep Learning": "Apply those architectures to real vision and language problems with transfer learning.",
  "Generative AI": "Understand and build autoencoders, GANs, diffusion models and the training of large language models.",
  "Advanced & Research": "Read, reproduce and extend research; understand self-supervision, RL, interpretability and robustness.",
  "Engineering & Deployment": "Implement in PyTorch/Keras, evaluate honestly, and ship models to production.",
};

export default function RoadmapPage() {
  return (
    <Container>
      <PageHeader eyebrow="Learning roadmap" title="Beginner to deep learning engineer">
        Follow the phases in order. Within each phase, modules are ordered by dependency — nothing assumes knowledge that hasn&apos;t been taught.
      </PageHeader>
      <ol className="relative mt-10 border-l border-border pl-8">
        {PHASES.map((ph, i) => {
          const mods = MODULES.filter((m) => m.phase === ph);
          return (
            <li key={ph} className="mb-12">
              <span className="absolute -left-[13px] flex h-6 w-6 items-center justify-center rounded-full border border-border-strong bg-bg font-mono text-xs">{i + 1}</span>
              <h2 className="font-serif text-2xl font-semibold">{ph}</h2>
              <p className="mt-1 text-muted">{PHASE_GOAL[ph]}</p>
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                {mods.map((m) => {
                  const full = m.topics.filter((t) => hasLesson(`${m.id}/${t.slug}`)).length;
                  return (
                    <Link key={m.id} href={`/curriculum#${m.id}`} className="card p-4 hover:border-border-strong">
                      <div className="flex items-baseline justify-between gap-2">
                        <span className="font-medium">
                          <span className="mr-2 font-mono text-xs text-faint">{String(m.number).padStart(2, "0")}</span>
                          {m.title}
                        </span>
                        <span className="shrink-0 font-mono text-xs text-faint">
                          {full}/{m.topics.length} published
                        </span>
                      </div>
                      <p className="mt-1 text-sm text-muted">{m.tagline}</p>
                    </Link>
                  );
                })}
              </div>
            </li>
          );
        })}
        <li>
          <span className="absolute -left-[13px] flex h-6 w-6 items-center justify-center rounded-full border border-accent bg-accent-soft font-mono text-xs">★</span>
          <h2 className="font-serif text-2xl font-semibold">Capstone projects</h2>
          <p className="mt-1 text-muted">Bring everything together in three end-to-end systems.</p>
          <div className="mt-4 grid gap-3 md:grid-cols-3">
            {PROJECTS.map((p) => (
              <Link key={p.id} href={`/projects#${p.id}`} className="card p-4 hover:border-border-strong">
                <div className="text-xs text-faint">{p.domain}</div>
                <div className="font-medium">{p.title}</div>
              </Link>
            ))}
          </div>
        </li>
      </ol>
    </Container>
  );
}
