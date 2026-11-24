import { MODULES } from "@/content/curriculum";
import { LESSONS } from "@/content/lessons";
import { GLOSSARY } from "@/content/glossary";
import { PAPERS } from "@/content/papers";
import { SIMS } from "@/components/sims/registry";
import { SECTION_ORDER } from "@/lib/types";
import { blocksToText, type SearchDoc } from "@/lib/search";

export const dynamic = "force-static";

/** The full-text search index, generated once at build time. */
export function GET() {
  const docs: SearchDoc[] = [];
  for (const m of MODULES) {
    docs.push({ kind: "module", title: `Module ${m.number}: ${m.title}`, href: `/curriculum#${m.id}`, context: m.phase, text: `${m.tagline} ${m.topics.map((t) => t.title).join(" ")}` });
    for (const t of m.topics) {
      docs.push({ kind: "topic", title: t.title, href: `/learn/${m.id}/${t.slug}`, context: `Module ${m.number} · ${m.title}`, text: `${t.summary} ${t.subtopics.join(" · ")}` });
    }
  }
  for (const l of LESSONS) {
    const [mid] = l.id.split("/");
    const mod = MODULES.find((m) => m.id === mid)!;
    for (const [key, label] of SECTION_ORDER) {
      const { prose, code, formulas } = blocksToText(l.sections[key] ?? []);
      const href = `/learn/${l.id}#${key}`;
      if (prose) docs.push({ kind: "section", title: `${l.title} — ${label}`, href, context: `Module ${mod.number}`, text: prose });
      code.forEach((c, i) => docs.push({ kind: "code", title: `${l.title} — code ${i + 1} (${label})`, href, context: "Code example", text: c }));
      if (formulas.length) docs.push({ kind: "formula", title: `${l.title} — formulas (${label})`, href, context: "Mathematics", text: formulas.join("  ·  ") });
    }
  }
  for (const g of GLOSSARY) docs.push({ kind: "glossary", title: g.term, href: `/glossary#${g.key}`, context: "Glossary", text: g.definition + (g.tex ? " " + g.tex : "") });
  for (const [id, s] of Object.entries(SIMS)) docs.push({ kind: "simulator", title: s.title, href: `/playground/${id}`, context: "Interactive simulator", text: s.blurb });
  for (const p of PAPERS) docs.push({ kind: "paper", title: p.title, href: `/papers#${p.key}`, context: `${p.authors} (${p.year})`, text: `${p.contribution} ${p.topics.join(" ")}` });
  return Response.json(docs);
}
