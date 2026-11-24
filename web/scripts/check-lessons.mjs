// Independent lesson verifier.
//   node scripts/check-lessons.mjs [lesson-file-substring] [--no-python]
// Checks every lesson in content/lessons/: 16 sections, interview/quiz/exercise counts, KaTeX strict render
// of every formula, internal links, and re-runs every Python block comparing stdout with its `output` field.
import { readdirSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import katex from "katex";

const root = path.resolve(import.meta.dirname, "..");
const PY = path.resolve(root, "..", ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
const SECTIONS = ["intro", "prerequisites", "intuition", "mechanism", "math", "visual", "scratch", "framework", "variations", "applications", "tradeoffs", "mistakes", "interview", "practice", "summary", "next"];
const filter = process.argv.slice(2).find((a) => !a.startsWith("--"));
const runPython = !process.argv.includes("--no-python");

// Topic ids from the curriculum (parse the t("slug", ...) calls per module id).
const curSrc = (await import("node:fs")).readFileSync(path.join(root, "content/curriculum.ts"), "utf8");
const topicIds = new Set();
for (const m of curSrc.matchAll(/id: "(m\d\d)"[\s\S]*?topics: \[([\s\S]*?)\n    \],/g)) for (const t of m[2].matchAll(/t\("([^"]+)"/g)) topicIds.add(`${m[1]}/${t[1]}`);

const tmp = mkdtempSync(path.join(tmpdir(), "lessoncheck-"));
let failures = 0;

function walk(blocks, fn) {
  for (const b of blocks ?? []) {
    fn(b);
    if (b.t === "details") walk(b.blocks, fn);
    if (b.t === "tabs") b.tabs.forEach((t) => walk(t.blocks, fn));
    if (b.t === "exercise") walk(b.solution, fn);
  }
}
const texts = (b) => {
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

const files = readdirSync(path.join(root, "content/lessons")).filter((f) => /^m\d\d-.*\.ts$/.test(f) && (!filter || f.includes(filter)));
for (const file of files) {
  const problems = [];
  const mod = await import(pathToFileURL(path.join(root, "content/lessons", file)).href);
  const L = mod.default;
  if (!topicIds.has(L.id)) problems.push(`id ${L.id} not in curriculum`);
  for (const s of SECTIONS) if (!L.sections?.[s]?.length) problems.push(`section '${s}' empty`);
  const qa = { beginner: 0, intermediate: 0, advanced: 0 };
  let quizQ = 0, exercises = 0, sims = 0, tex = 0, py = 0, pyOk = 0;
  for (const s of SECTIONS) {
    walk(L.sections[s], (b) => {
      if (b.t === "qa") qa[b.level]++;
      if (b.t === "quiz") quizQ += b.questions.length;
      if (b.t === "exercise") exercises++;
      if (b.t === "sim") sims++;
      const formulas = [];
      if (b.t === "math") formulas.push(b.tex);
      if (b.t === "steps") b.items.forEach((it) => it.tex && formulas.push(it.tex));
      for (const t of texts(b)) {
        for (const m of t.matchAll(/(?<!\\)\$([^$]+)\$/g)) formulas.push(m[1]);
        if ((t.replace(/\\\$/g, "").match(/\$/g) ?? []).length % 2) problems.push(`[${s}] unbalanced $ in: ${t.slice(0, 70)}`);
        for (const m of t.matchAll(/\]\((\/learn\/[^)#]+)/g)) if (!topicIds.has(m[1].replace("/learn/", ""))) problems.push(`[${s}] bad link ${m[1]}`);
      }
      for (const f of formulas) {
        tex++;
        try {
          katex.renderToString(f, { throwOnError: true, strict: "error" });
        } catch (e) {
          problems.push(`[${s}] KaTeX: ${String(e.message).slice(0, 90)} :: ${f.slice(0, 60)}`);
        }
      }
      if (runPython && b.t === "code" && b.lang === "python" && b.output && !/not executed|vary between runs/i.test(b.title ?? "") && !/^\s*(import|from)\s+(tensorflow|keras)\b/m.test(b.code)) {
        py++;
        const f = path.join(tmp, `${file}-${py}.py`);
        writeFileSync(f, b.code);
        const r = spawnSync(PY, [f], { encoding: "utf8", timeout: 240000, /* generous: other jobs can load the CPU */ cwd: tmp, env: { ...process.env, MPLBACKEND: "Agg", PYTHONIOENCODING: "utf-8" } });
        const norm = (x) => (x ?? "").replace(/\r/g, "").split("\n").map((l) => l.trimEnd()).join("\n").trim();
        if (r.status !== 0) problems.push(`[${s}] python block ${py} (${b.title ?? ""}) exited ${r.status}: ${norm(r.stderr).split("\n").slice(-1)[0]}`);
        else if (norm(r.stdout) !== norm(b.output)) {
          const a = norm(r.stdout).split("\n"), e = norm(b.output).split("\n");
          const i = a.findIndex((l, k) => l !== e[k]);
          problems.push(`[${s}] python block ${py} (${b.title ?? ""}) output differs at line ${i + 1}: got "${a[i] ?? "<eof>"}" expected "${e[i] ?? "<eof>"}"`);
        } else pyOk++;
      }
    });
  }
  if (qa.beginner < 3 || qa.intermediate < 3 || qa.advanced < 2) problems.push(`interview counts ${JSON.stringify(qa)}`);
  if (quizQ < 6) problems.push(`only ${quizQ} quiz questions`);
  if (exercises < 2) problems.push(`only ${exercises} exercises`);
  if (sims < 1) problems.push(`no simulator embedded`);
  const status = problems.length ? "FAIL" : "ok  ";
  if (problems.length) failures++;
  console.log(`${status} ${file}  tex=${tex} python=${pyOk}/${py} quiz=${quizQ} ex=${exercises} qa=${qa.beginner}/${qa.intermediate}/${qa.advanced}`);
  for (const p of problems) console.log("     - " + p);
}
console.log(failures ? `\n${failures} lesson(s) with problems` : "\nall lessons pass");
process.exit(failures ? 1 : 0);
