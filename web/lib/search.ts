import type { Block } from "./types";

export type DocKind = "module" | "topic" | "section" | "code" | "formula" | "glossary" | "simulator" | "paper";

export interface SearchDoc {
  kind: DocKind;
  title: string;
  href: string;
  context: string; // breadcrumb-ish label
  text: string; // searchable body
}

export interface SearchHit {
  doc: SearchDoc;
  score: number;
  snippet: string;
}

/** Flatten a block tree into plain text (LaTeX kept, inline markup stripped). */
export function blocksToText(blocks: Block[]): { prose: string; code: string[]; formulas: string[] } {
  const prose: string[] = [];
  const code: string[] = [];
  const formulas: string[] = [];
  const strip = (s: string) => s.replace(/\*\*|`/g, "").replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  const walk = (bs: Block[]) => {
    for (const b of bs) {
      switch (b.t) {
        case "p":
        case "h":
          prose.push(strip(b.text));
          break;
        case "callout":
          prose.push(strip(b.text));
          break;
        case "list":
          prose.push(...b.items.map(strip));
          break;
        case "math":
          formulas.push(b.tex + (b.caption ? " — " + strip(b.caption) : ""));
          break;
        case "code":
          code.push((b.title ? b.title + "\n" : "") + b.code);
          break;
        case "table":
          prose.push(b.head.join(" "), ...b.rows.map((r) => strip(r.join(" "))));
          break;
        case "steps":
          prose.push(...b.items.map((s) => strip(`${s.title} ${s.text ?? ""}`)));
          break;
        case "details":
          walk(b.blocks);
          break;
        case "tabs":
          b.tabs.forEach((t) => walk(t.blocks));
          break;
        case "qa":
          prose.push(strip(b.q), strip(b.a));
          break;
        case "exercise":
          prose.push(strip(b.prompt));
          walk(b.solution);
          break;
        case "quiz":
          prose.push(...b.questions.map((q) => strip(q.q)));
          break;
        default:
          break;
      }
    }
  };
  walk(blocks);
  return { prose: prose.join(" "), code, formulas };
}

const norm = (s: string) => s.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "");

export function searchDocs(docs: SearchDoc[], query: string, limit = 30): SearchHit[] {
  const q = norm(query.trim());
  if (q.length < 2) return [];
  const terms = q.split(/\s+/).filter(Boolean);
  const KIND_BOOST: Record<DocKind, number> = { topic: 3, module: 2.5, glossary: 2.5, simulator: 2, section: 1, formula: 0.9, code: 0.8, paper: 1.5 };
  const hits: SearchHit[] = [];
  for (const doc of docs) {
    const title = norm(doc.title);
    const body = norm(doc.text);
    let score = 0;
    let allFound = true;
    for (const term of terms) {
      const inTitle = title.includes(term);
      const idx = body.indexOf(term);
      if (!inTitle && idx < 0) {
        allFound = false;
        break;
      }
      if (inTitle) score += title.startsWith(term) ? 12 : 8;
      if (idx >= 0) {
        // count occurrences, capped
        let c = 0, p = idx;
        while (p >= 0 && c < 6) {
          c++;
          p = body.indexOf(term, p + term.length);
        }
        score += 1 + c;
      }
    }
    if (!allFound) continue;
    if (title === q) score += 20;
    if (title.includes(q)) score += 6;
    score *= KIND_BOOST[doc.kind];
    hits.push({ doc, score, snippet: makeSnippet(doc.text, terms[0]) });
  }
  return hits.sort((a, b) => b.score - a.score).slice(0, limit);
}

function makeSnippet(text: string, term: string) {
  const i = norm(text).indexOf(term);
  if (i < 0) return text.slice(0, 140);
  const start = Math.max(0, i - 60);
  return (start > 0 ? "…" : "") + text.slice(start, i + 100).replace(/\s+/g, " ") + "…";
}
