import { Info, Lightbulb, AlertTriangle, Sparkles, KeyRound } from "lucide-react";
import type { Block } from "@/lib/types";
import { Inline, MathBlock, renderTex } from "./Inline";
import { CodeBlock } from "./CodeBlock";
import { Quiz } from "./Quiz";
import { Details, Exercise, Tabs } from "./Interactive";
import { SimEmbed } from "@/components/sims/SimEmbed";

const CALLOUT = {
  note: { icon: Info, label: "Note", cls: "border-border bg-surface-2" },
  tip: { icon: Lightbulb, label: "Tip", cls: "border-accent/30 bg-accent-soft" },
  warning: { icon: AlertTriangle, label: "Watch out", cls: "border-clay/40 bg-clay-soft" },
  analogy: { icon: Sparkles, label: "Analogy", cls: "border-gold/40 bg-gold-soft" },
  key: { icon: KeyRound, label: "Key idea", cls: "border-accent/40 bg-accent-soft" },
} as const;

const LEVEL_CLS = {
  beginner: "text-accent",
  intermediate: "text-gold",
  advanced: "text-clay",
};

export function Blocks({ blocks, ctx }: { blocks: Block[]; ctx: string }) {
  return (
    <>
      {blocks.map((b, i) => (
        <BlockView key={i} b={b} ctx={`${ctx}-${i}`} />
      ))}
    </>
  );
}

function BlockView({ b, ctx }: { b: Block; ctx: string }) {
  switch (b.t) {
    case "p":
      return (
        <p>
          <Inline text={b.text} />
        </p>
      );
    case "h":
      return (
        <h3 className="mt-7 mb-2 font-serif text-lg font-semibold">
          <Inline text={b.text} />
        </h3>
      );
    case "math":
      return <MathBlock tex={b.tex} caption={b.caption} />;
    case "list": {
      const L = b.ordered ? "ol" : "ul";
      return (
        <L>
          {b.items.map((it, i) => (
            <li key={i}>
              <Inline text={it} />
            </li>
          ))}
        </L>
      );
    }
    case "code":
      return <CodeBlock code={b.code} lang={b.lang} title={b.title} output={b.output} runnable={b.runnable ?? b.lang === "python"} />;
    case "callout": {
      const c = CALLOUT[b.kind];
      const Icon = c.icon;
      return (
        <div className={`my-5 flex gap-3 rounded-lg border p-4 ${c.cls}`}>
          <Icon size={18} className="mt-1 shrink-0 text-muted" />
          <div className="min-w-0">
            <div className="text-sm font-semibold">{b.title ?? c.label}</div>
            <div className="text-[0.95rem] [&>p]:my-1">
              <Inline text={b.text} />
            </div>
          </div>
        </div>
      );
    }
    case "table":
      return (
        <div className="my-5 overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead className="bg-surface-2">
              <tr>
                {b.head.map((h, i) => (
                  <th key={i} className="px-3 py-2 text-left font-semibold">
                    <Inline text={h} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {b.rows.map((r, i) => (
                <tr key={i} className="border-t border-border align-top">
                  {r.map((c, j) => (
                    <td key={j} className="px-3 py-2">
                      <Inline text={c} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    case "sim":
      return <SimEmbed id={b.id} caption={b.caption} />;
    case "steps":
      return (
        <ol className="!ml-0 my-5 !list-none space-y-3">
          {b.items.map((s, i) => (
            <li key={i} className="flex gap-3">
              <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-border-strong text-xs text-muted">{i + 1}</span>
              <div className="min-w-0 flex-1">
                <div className="font-medium">
                  <Inline text={s.title} />
                </div>
                {s.text && (
                  <div className="text-[0.95rem] text-muted">
                    <Inline text={s.text} />
                  </div>
                )}
                {s.tex && <div className="mt-1 overflow-x-auto" dangerouslySetInnerHTML={{ __html: renderTex(s.tex, true) }} />}
              </div>
            </li>
          ))}
        </ol>
      );
    case "details":
      return (
        <Details summary={b.summary}>
          <Blocks blocks={b.blocks} ctx={ctx} />
        </Details>
      );
    case "tabs":
      return (
        <Tabs labels={b.tabs.map((t) => t.label)}>
          {b.tabs.map((t, i) => (
            <div key={i}>
              <Blocks blocks={t.blocks} ctx={`${ctx}-${i}`} />
            </div>
          ))}
        </Tabs>
      );
    case "qa":
      return (
        <Details summary={`[${b.level}] ${b.q.replace(/\$[^$]*\$/g, (m) => m.slice(1, -1))}`}>
          <div className="py-2">
            <span className={`text-xs font-medium uppercase tracking-wider ${LEVEL_CLS[b.level]}`}>{b.level}</span>
            <p>
              <Inline text={b.a} />
            </p>
          </div>
        </Details>
      );
    case "quiz":
      return <Quiz questions={b.questions} quizKey={ctx} />;
    case "exercise":
      return (
        <Exercise title={b.title} prompt={<Inline text={b.prompt} />} hint={b.hint ? <Inline text={b.hint} /> : undefined}>
          <Blocks blocks={b.solution} ctx={ctx} />
        </Exercise>
      );
  }
}
