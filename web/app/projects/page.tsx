import type { Metadata } from "next";
import Link from "next/link";
import { PROJECTS } from "@/content/projects";
import { getTopic } from "@/content/curriculum";
import { Container, PageHeader } from "@/components/pages/PageHeader";
import { MilestoneList } from "@/components/pages/MilestoneList";

export const metadata: Metadata = { title: "Industry Projects" };

export default function ProjectsPage() {
  return (
    <Container>
      <PageHeader eyebrow="Capstone projects" title="Three complete deep learning systems">
        Real projects, built the way companies build them: requirements, design, data, choosing a model, training, testing, an API, a frontend and deployment. Each one runs a real model — no fake, pre-made outputs.
      </PageHeader>
      <div className="my-8 rounded-lg border border-gold/40 bg-gold-soft p-4 text-sm">
        <strong>Status:</strong> the plans below are final. Step-by-step build guides and source code come next (see <code className="font-mono text-xs">docs/ROADMAP.md</code>). Milestones you tick here are saved to your progress.
      </div>
      {PROJECTS.map((p, i) => (
        <article key={p.id} id={p.id} className="scroll-mt-20 border-b border-border py-12 last:border-0">
          <div className="font-mono text-sm text-faint">Project {i + 1} · {p.domain}</div>
          <h2 className="mt-1 font-serif text-3xl font-semibold">{p.title}</h2>
          <p className="mt-3 max-w-3xl text-lg text-muted">{p.problem}</p>

          <div className="mt-8 grid gap-8 lg:grid-cols-2 [&>*]:min-w-0">
            <div>
              <H>Use cases</H>
              <ul className="list-disc space-y-1 pl-5 text-sm">{p.useCases.map((u) => <li key={u}>{u}</li>)}</ul>
              <H>What it must do</H>
              <ul className="list-disc space-y-1 pl-5 text-sm">{p.requirements.functional.map((u) => <li key={u}>{u}</li>)}</ul>
              <H>Quality targets (non-functional)</H>
              <ul className="list-disc space-y-1 pl-5 text-sm">{p.requirements.nonFunctional.map((u) => <li key={u}>{u}</li>)}</ul>
            </div>
            <div>
              <H>Data flow</H>
              <ol className="space-y-2 text-sm">
                {p.architecture.map((a, k) => (
                  <li key={a} className="flex gap-3">
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-border-strong font-mono text-[0.65rem] text-muted">{k + 1}</span>
                    {a}
                  </li>
                ))}
              </ol>
            </div>
          </div>

          <H>Technology choices</H>
          <div className="overflow-x-auto rounded-lg border border-border">
            <table className="w-full text-sm">
              <tbody>
                {p.stack.map((s) => (
                  <tr key={s.layer} className="border-b border-border last:border-0 align-top">
                    <td className="w-32 px-3 py-2 font-medium">{s.layer}</td>
                    <td className="px-3 py-2">{s.choice}</td>
                    <td className="px-3 py-2 text-muted">{s.why}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-8 grid gap-8 lg:grid-cols-2 [&>*]:min-w-0">
            <div>
              <H>Data</H>
              <ul className="list-disc space-y-1 pl-5 text-sm">{p.data.map((u) => <li key={u}>{u}</li>)}</ul>
              <H>Models compared</H>
              <ul className="space-y-2 text-sm">
                {p.models.map((m) => (
                  <li key={m.name}>
                    <span className="font-medium">{m.name}</span> — <span className="text-muted">{m.why}</span>
                  </li>
                ))}
              </ul>
              <H>Hardware</H>
              <p className="text-sm text-muted">{p.hardware}</p>
              <H>Lessons to take first</H>
              <div className="flex flex-wrap gap-2">
                {p.prerequisites.map((t) => (
                  <Link key={t} href={`/learn/${t}`} className="rounded-full border border-border px-2.5 py-1 text-xs hover:border-accent">
                    {getTopic(t)?.topic.title ?? t}
                  </Link>
                ))}
              </div>
            </div>
            <div>
              <H>Folder structure</H>
              <pre className="overflow-x-auto rounded-lg border border-border bg-code p-4 font-mono text-xs leading-relaxed">{p.tree}</pre>
              <H>Milestones</H>
              <MilestoneList projectId={p.id} milestones={p.milestones} />
            </div>
          </div>
        </article>
      ))}
    </Container>
  );
}

function H({ children }: { children: React.ReactNode }) {
  return <h3 className="mt-6 mb-2 text-xs font-medium uppercase tracking-wider text-faint">{children}</h3>;
}
