import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { SIMS } from "@/components/sims/registry";
import { SimEmbed } from "@/components/sims/SimEmbed";
import { getTopic } from "@/content/curriculum";
import type { SimId } from "@/lib/types";

export function generateStaticParams() {
  return Object.keys(SIMS).map((sim) => ({ sim }));
}

export async function generateMetadata({ params }: { params: Promise<{ sim: string }> }): Promise<Metadata> {
  const { sim } = await params;
  const s = SIMS[sim as SimId];
  return s ? { title: s.title, description: s.blurb } : {};
}

export default async function SimPage({ params }: { params: Promise<{ sim: string }> }) {
  const { sim } = await params;
  const s = SIMS[sim as SimId];
  if (!s) notFound();
  const lesson = getTopic(s.lesson);
  return (
    <div className="mx-auto max-w-[1300px] px-4 py-8">
      <nav className="mb-3 text-xs text-faint">
        <Link href="/playground" className="hover:text-text">
          Playground
        </Link>{" "}
        / <span className="text-muted">{s.title}</span>
      </nav>
      <h1 className="font-serif text-3xl font-semibold">{s.title}</h1>
      <p className="mt-2 max-w-3xl text-muted">{s.blurb}</p>
      <SimEmbed id={sim as SimId} />
      {lesson && (
        <p className="text-sm text-muted">
          You can learn the ideas behind this simulator in{" "}
          <Link href={`/learn/${s.lesson}`} className="text-accent underline underline-offset-2">
            {lesson.topic.title}
          </Link>
          .
        </p>
      )}
    </div>
  );
}
