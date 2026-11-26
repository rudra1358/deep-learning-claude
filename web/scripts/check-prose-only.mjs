// Guard for prose rewrites.
//   node scripts/check-prose-only.mjs [lesson-file-substring] [--base=<git-ref>]
// Compares every lesson in content/lessons/ with the same file at a git ref (default: HEAD) and fails if anything
// other than explanatory prose changed. Prose (p, h, list items, callouts, table cells, step text, qa answers, quiz
// wording, exercise prompts/hints, captions, summaries) may be rewritten and new prose blocks may be added. These
// must stay identical: lesson id/level/prerequisites/next/terms, every code block (lang, title, code, output),
// every display formula, step formulas, simulator embeds (order), quiz structure and answers, the number of
// exercises and interview questions per level, and the set of /learn/ links (links may be added, not removed).
import { readdirSync, writeFileSync, mkdtempSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = path.resolve(import.meta.dirname, "..");
const filter = process.argv.slice(2).find((a) => !a.startsWith("--"));
const base = (process.argv.find((a) => a.startsWith("--base=")) ?? "--base=HEAD").slice(7);
const tmp = mkdtempSync(path.join(tmpdir(), "prosecheck-"));
const SECTIONS = ["intro", "prerequisites", "intuition", "mechanism", "math", "visual", "scratch", "framework", "variations", "applications", "tradeoffs", "mistakes", "interview", "practice", "summary", "next"];

function walk(blocks, fn) {
  for (const b of blocks ?? []) {
    fn(b);
    if (b.t === "details") walk(b.blocks, fn);
    if (b.t === "tabs") b.tabs.forEach((t) => walk(t.blocks, fn));
    if (b.t === "exercise") walk(b.solution, fn);
  }
}
const proseOf = (b) => {
  const out = [];
  if (b.t === "p" || b.t === "h" || b.t === "callout") out.push(b.text);
  if (b.t === "list") out.push(...b.items);
  if (b.t === "table") out.push(...b.head, ...b.rows.flat());
  if (b.t === "steps") b.items.forEach((s) => out.push(s.title, s.text ?? ""));
  if (b.t === "qa") out.push(b.q, b.a);
  if (b.t === "exercise") out.push(b.prompt, b.hint ?? "");
  if (b.t === "quiz") b.questions.forEach((q) => out.push(q.q, q.explain, ...(q.options ?? [])));
  if ((b.t === "math" || b.t === "sim") && b.caption) out.push(b.caption);
  return out;
};

/** Everything that must not change, as a list of strings per section. */
function invariants(L) {
  const inv = { meta: JSON.stringify([L.id, L.level, L.prerequisites, L.next, L.terms ?? []]) };
  const links = new Set();
  for (const s of SECTIONS) {
    const items = [];
    const qa = { beginner: 0, intermediate: 0, advanced: 0 };
    let exercises = 0;
    walk(L.sections[s], (b) => {
      if (b.t === "code") items.push("code " + JSON.stringify([b.lang, b.title ?? "", b.code, b.output ?? "", !!b.runnable]));
      if (b.t === "math") items.push("math " + b.tex);
      if (b.t === "sim") items.push("sim " + b.id);
      if (b.t === "steps") b.items.forEach((it) => it.tex && items.push("steptex " + it.tex));
      if (b.t === "tabs") items.push("tabs " + b.tabs.map((t) => t.label).join("|"));
      if (b.t === "quiz") b.questions.forEach((q) => items.push("quiz " + JSON.stringify(q.kind === "numeric" ? ["n", q.answer, q.tolerance] : ["c", q.answer, q.options.length])));
      if (b.t === "qa") qa[b.level]++;
      if (b.t === "exercise") exercises++;
      for (const t of proseOf(b)) for (const m of t.matchAll(/\]\((\/[^)]+)\)/g)) links.add(m[1]);
    });
    inv[s] = { items, qa, exercises };
  }
  inv.links = links;
  return inv;
}

const files = readdirSync(path.join(root, "content/lessons")).filter((f) => /^m\d\d-.*\.ts$/.test(f) && (!filter || f.includes(filter)));
let failures = 0, changed = 0;
for (const file of files) {
  let old;
  try {
    old = execFileSync("git", ["show", `${base}:web/content/lessons/${file}`], { cwd: root, encoding: "utf8", maxBuffer: 64 << 20 });
  } catch {
    console.log(`new  ${file}  (not in ${base}; nothing to compare)`);
    continue;
  }
  const oldPath = path.join(tmp, file);
  writeFileSync(oldPath, old);
  const A = (await import(pathToFileURL(oldPath).href)).default;
  const B = (await import(pathToFileURL(path.join(root, "content/lessons", file)).href)).default;
  const a = invariants(A), b = invariants(B);
  const problems = [];
  if (a.meta !== b.meta) problems.push("id/level/prerequisites/next/terms changed");
  for (const s of SECTIONS) {
    const x = a[s], y = b[s];
    if (!B.sections[s]?.length) problems.push(`[${s}] section missing or empty`);
    const n = Math.max(x.items.length, y.items.length);
    for (let i = 0; i < n; i++) {
      if (x.items[i] !== y.items[i]) {
        problems.push(`[${s}] protected item ${i + 1} differs: was "${(x.items[i] ?? "<none>").slice(0, 70)}" now "${(y.items[i] ?? "<none>").slice(0, 70)}"`);
        break;
      }
    }
    for (const lv of ["beginner", "intermediate", "advanced"]) if (y.qa[lv] < x.qa[lv]) problems.push(`[${s}] fewer ${lv} interview questions (${x.qa[lv]} → ${y.qa[lv]})`);
    if (y.exercises !== x.exercises) problems.push(`[${s}] exercise count ${x.exercises} → ${y.exercises}`);
  }
  for (const l of a.links) if (!b.links.has(l)) problems.push(`link removed: ${l}`);
  // Length budget: rewritten prose must not be longer than the original (5% slack for tiny lessons).
  const words = (L) => {
    let n = L.summary.split(/\s+/).length;
    for (const s of SECTIONS) walk(L.sections[s], (blk) => { for (const t of proseOf(blk)) n += t.split(/\s+/).filter(Boolean).length; if (blk.t === "details") n += blk.summary.split(/\s+/).length; if (blk.t === "exercise") n += blk.title.split(/\s+/).length; if (blk.t === "callout" && blk.title) n += blk.title.split(/\s+/).length; });
    return n;
  };
  const wa = words(A), wb = words(B);
  if (wb > wa * 1.05) problems.push(`prose grew: ${wa} → ${wb} words (${Math.round((wb / wa) * 100)}%); budget is ≤ 100% of the original`);
  const proseA = JSON.stringify(A.sections), proseB = JSON.stringify(B.sections);
  const didChange = proseA !== proseB || A.summary !== B.summary || A.title !== B.title;
  if (didChange) changed++;
  if (problems.length) failures++;
  console.log(`${problems.length ? "FAIL" : didChange ? "ok  " : "same"} ${file}  words ${wa} → ${wb} (${Math.round((wb / wa) * 100)}%)`);
  for (const p of problems) console.log("     - " + p);
}
console.log(`\n${changed} rewritten, ${failures} with problems (compared against ${base})`);
process.exit(failures ? 1 : 0);
