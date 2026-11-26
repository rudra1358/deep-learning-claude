// Finds formulas whose LaTeX backslashes were silently eaten (a plain "..." string instead of String.raw):
// e.g. "\sqrt" becomes "sqrt", "\text" becomes a TAB + "ext", "\frac" becomes a form-feed + "rac".
//   node scripts/scan-lost-backslashes.mjs
import { readdirSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = path.resolve(import.meta.dirname, "..");
const CMDS = "frac|dfrac|tfrac|sqrt|mathrm|mathbf|mathcal|mathbb|operatorname|text|textbf|sum|prod|int|lVert|rVert|lvert|rvert|cdot|cdots|ldots|times|alpha|beta|gamma|delta|epsilon|varepsilon|theta|lambda|sigma|mu|eta|tau|phi|psi|omega|Delta|Sigma|hat|tilde|bar|vec|log|exp|partial|nabla|infty|leq|geq|neq|approx|odot|otimes|top|max|min|arg|left|right|quad|in|to|mid|propto|sim";
const bare = new RegExp(`(?<![\\\\a-zA-Z])(${CMDS})(?=[^a-zA-Z])`, "g");
const files = readdirSync(path.join(root, "content/lessons")).filter((f) => /^m\d\d-.*\.ts$/.test(f));
let hits = 0;

function strings(x, out) {
  if (typeof x === "string") out.push(x);
  else if (Array.isArray(x)) x.forEach((y) => strings(y, out));
  else if (x && typeof x === "object") for (const [k, v] of Object.entries(x)) if (k !== "code" && k !== "output") strings(v, out);
  return out;
}

for (const f of files) {
  const L = (await import(pathToFileURL(path.join(root, "content/lessons", f)).href)).default;
  for (const s of strings(L, [])) {
    const problems = [];
    if (/[\t\b\f\v]/.test(s)) problems.push("control character (eaten \\t, \\b, \\f or \\v)");
    for (const m of s.matchAll(/(?<!\\)\$([^$]+)\$/g)) {
      const math = m[1];
      const found = [...math.matchAll(bare)].map((x) => x[1]).filter((c) => !(c.length <= 3 && /^[a-z]+$/.test(c) && !["exp", "log", "max", "min", "arg", "sum", "top", "bar", "hat", "vec", "int"].includes(c)));
      if (found.length) problems.push(`bare command(s) in $…$: ${[...new Set(found)].join(", ")}`);
    }
    if (problems.length) {
      hits++;
      console.log(`${f}: ${problems.join("; ")}\n    ${s.replace(/\s+/g, " ").slice(0, 140)}`);
    }
  }
}
console.log(hits ? `\n${hits} suspicious string(s)` : "no lost backslashes found");
