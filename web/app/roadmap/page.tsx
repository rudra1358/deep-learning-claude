import type { Metadata } from "next";
import Link from "next/link";
import { MODULES, PHASES } from "@/content/curriculum";
import { hasLesson } from "@/content/lessons";
import { PROJECTS } from "@/content/projects";
import { Container, PageHeader } from "@/components/pages/PageHeader";

export const metadata: Metadata = { title: "Learning Roadmap" };

const PHASE_GOAL: Record<string, string> = {
  Foundations: "Get comfortable reading and writing the math of deep learning, and computing it in NumPy.",
  "Neural Networks": "Work out and build a full neural network from scratch — forward pass, loss, backpropagation and optimizer.",
  "Core Architectures": "Understand CNNs, RNNs/LSTMs and Transformers well enough to build each one yourself.",
  "Applied Deep Learning": "Use those models on real image and language problems, reusing pretrained models (transfer learning).",
  "Generative AI": "Understand and build autoencoders, GANs and diffusion models, and see how large language models are trained.",
  "Advanced & Research": "Read, reproduce and build on research. Learn self-supervised learning, RL, interpretability and robustness.",
  "Engineering & Deployment": "Build in PyTorch/Keras, measure your models honestly, and put them into real use.",
};

export default function RoadmapPage() {
  return (
    <Container>
      <PageHeader eyebrow="Learning roadmap" title="From beginner to deep learning engineer">
        Follow the phases in order. Inside each phase, modules are ordered so that nothing expects something you haven&apos;t learned yet.
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
                          {full}/{m.topics.length} ready
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
          <p className="mt-1 text-muted">Put everything together in three complete, real-world systems.</p>
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
