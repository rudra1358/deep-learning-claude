import type { Metadata } from "next";
import Link from "next/link";
import { getModule } from "@/content/curriculum";
import { hasLesson } from "@/content/lessons";
import { Container, PageHeader } from "@/components/pages/PageHeader";
import { SimEmbed } from "@/components/sims/SimEmbed";
import { MathBlock } from "@/components/content/Inline";

export const metadata: Metadata = { title: "Mathematics Lab" };

const CORE = [
  { tex: String.raw`z = Wx + b`, label: "A dense layer is a matrix times a vector, plus a bias", href: "/learn/m02/vectors-matrices" },
  { tex: String.raw`\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a}\,\frac{\partial a}{\partial z}\,\frac{\partial z}{\partial w}`, label: "Backpropagation is the chain rule", href: "/learn/m02/derivatives-chain-rule" },
  { tex: String.raw`\theta_{t+1} = \theta_t - \eta\,\nabla_\theta L(\theta_t)`, label: "Gradient descent", href: "/learn/m06/gradient-descent" },
  { tex: String.raw`H(p, q) = -\sum_k p_k \log q_k`, label: "Cross-entropy: the classification loss", href: "/learn/m05/classification-losses" },
  { tex: String.raw`\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V`, label: "Scaled dot-product attention", href: "/learn/m10/scaled-dot-product-attention" },
];

export default function MathLabPage() {
  const math = getModule("m02")!;
  return (
    <Container>
      <PageHeader eyebrow="Math lab" title="The math, made hands-on">
        Linear algebra, calculus, probability and optimization — each with an experiment you can play with. Five equations do most of the work in deep learning. Each one links to the lesson that explains it.
      </PageHeader>
      <section className="grid grid-cols-1 gap-4 py-10 md:grid-cols-2">
        {CORE.map((c) => (
          <Link key={c.href} href={c.href} className="card p-5 hover:border-border-strong">
            <MathBlock tex={c.tex} />
            <div className="text-sm text-muted">{c.label}</div>
          </Link>
        ))}
      </section>
      <section className="py-6">
        <h2 className="mb-4 font-serif text-2xl font-semibold">Experiments</h2>
        <SimEmbed id="matmul" />
        <SimEmbed id="chain-rule" />
        <SimEmbed id="gradient-descent" />
      </section>
      <section className="py-6">
        <h2 className="mb-4 font-serif text-2xl font-semibold">Module 2 lessons</h2>
        <ul className="grid gap-2 sm:grid-cols-2">
          {math.topics.map((t) => (
            <li key={t.slug}>
              <Link href={`/learn/m02/${t.slug}`} className="flex items-center justify-between rounded-lg border border-border bg-surface px-4 py-3 text-sm hover:border-border-strong">
                {t.title}
                {hasLesson(`m02/${t.slug}`) && <span className="rounded bg-accent-soft px-1.5 py-0.5 text-[0.68rem] text-accent-strong">full lesson</span>}
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </Container>
  );
}
