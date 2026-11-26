# Content & Simulator Authoring Guide

This is the contract every lesson and simulator in the platform follows. Read it fully before writing content.

## 1. Lessons

A lesson is a TypeScript module in `web/content/lessons/<moduleId>-<slug>.ts` that default-exports a `Lesson`
(schema: `web/lib/types.ts`). Its `id` must equal `"<moduleId>/<slug>"` of a topic in `web/content/curriculum.ts`.

```ts
import type { Lesson } from "@/lib/types";
const lesson: Lesson = { id: "m03/artificial-neuron", title, summary, minutes, level, prerequisites: [...], next: [...], terms: [...], sections: { intro: [...], ... } };
export default lesson;
```

### All 16 sections are mandatory (keys of `sections`)

| key | Must contain |
|---|---|
| `intro` | What it is, why it was introduced, what problem it solves, where it is used. |
| `prerequisites` | List of prior concepts with links (`[Derivatives](/learn/m02/derivatives-chain-rule)`). Say what specifically is needed from each. |
| `intuition` | A real-world analogy (`callout` kind `analogy`) + plain-language explanation for a first-time learner. |
| `mechanism` | Full step-by-step internal mechanism. Do not skip intermediate operations. `steps` blocks are good here. |
| `math` | Formulas, **every variable defined**, derivation(s), a fully worked numerical example with every intermediate number. |
| `visual` | One or more `{ t: "sim", id }` embeds + text telling the learner exactly what to try and what to observe. |
| `scratch` | Complete, executable **Python + NumPy** implementation: imports, sample data, execution, printed output, comments, no unexplained magic numbers, and an `output` field containing the exact output from actually running it. |
| `framework` | `tabs` block: "PyTorch" tab (always) and "TensorFlow / Keras" tab (where appropriate). Complete programs with output. |
| `variations` | Subtypes/variants with a comparison `table` (advantages, disadvantages, use cases). |
| `applications` | Concrete industry uses (name real, well-known systems only where you are sure). |
| `tradeoffs` | Strengths, weaknesses, practical trade-offs. |
| `mistakes` | Common implementation errors and misconceptions — each with *why* it is wrong and the fix. |
| `interview` | `qa` blocks: ≥3 beginner, ≥3 intermediate, ≥2 advanced, with full answers. |
| `practice` | One `quiz` block (≥6 questions, mixing `choice` and `kind: "numeric"`), plus ≥2 `exercise` blocks (one numerical, one coding) with full solutions. |
| `summary` | Structured revision list of key takeaways and key formulas. |
| `next` | What to learn next and *why* (link with `/learn/...`). |

### Inline text dialect

Inside every text string: `**bold**`, `` `code` ``, `$LaTeX$`, `[text](/learn/m04/backpropagation)`.
**Use `String.raw` template literals for every string containing a backslash** (LaTeX), e.g.
`{ t: "math", tex: String.raw`\frac{\partial L}{\partial w}` }`. Avoid literal backticks inside String.raw strings;
for inline code inside a String.raw string you can't use backticks — split the paragraph or use a normal string.
Python code also goes in `String.raw` (so `\n` stays literal). Do not use `${` inside template literals.

### Accuracy rules (non-negotiable)

- Every Python program must be **actually executed** before being included; paste its real output in `output`.
  Verification interpreter (has numpy, matplotlib, scikit-learn, torch CPU):
  `"C:\Users\rudra\Documents\Deep Learning\.venv\Scripts\python.exe"`. Write scratch files to your own temp dir, not the repo.
  Use fixed seeds (`np.random.default_rng(0)`, `torch.manual_seed(0)`) so output is reproducible.
  Keep runtimes short (< ~20 s on CPU).
- TensorFlow is not installable in the verification environment. TensorFlow/Keras code must use only stable, well-known
  public APIs; set its `output` to a description of what is printed (e.g. "Prints the loss for each of 5 epochs…")
  rather than invented numbers, and label the code title `"… (not executed in CI)"`.
- Never invent papers, citations, benchmarks or numbers. Cite original papers by authors + year + title only when certain.
- Numerical examples in `math` must be arithmetically correct — compute them with Python, don't do them in your head.
- Explain every new term before relying on it.
- No filler, no marketing tone. University-level depth, written clearly.

### Lessons learned from round 1 (follow these)

- Exemplar lessons to match in depth and tone: `web/content/lessons/m04-backpropagation.ts`, `m08-convolution.ts`.
- `*italic*` is supported; a `*` inside a `**bold**` span breaks the bold — avoid it.
- Do not edit `glossary.ts`, `index.ts`, `registry.ts`, `SimEmbed.tsx` or `types.ts`. Put glossary keys you would
  like added in your final report (key, term, one-sentence definition, optional TeX) and the integrator adds them.
- Verify mechanically: (a) load the lesson and check all 16 sections + interview/quiz counts, (b) run every
  `python` code block and diff against its `output`, (c) render every TeX string with KaTeX in strict mode,
  (d) check every `/learn/...` link exists in `curriculum.ts`.
- Link to lessons that already exist where relevant (see `web/content/lessons/index.ts`) rather than re-teaching
  them; it is fine to link to topics that are not written yet.
- **Run the project checker before reporting**: from `web/`, `node scripts/check-lessons.mjs <file-substring>` must print
  `ok` (it re-runs every Python block and needs byte-identical output). Mark nondeterministic blocks with
  "(… vary between runs)" in the title only if determinism is genuinely impossible.
- Put scratch files in a **uniquely named subfolder** of your temp/scratchpad dir (e.g. `scratch/m09-bptt/`) — authors
  running in parallel share the scratchpad and generic names collide.
- Bash heredocs in this Windows environment can mangle backslashes — write generator/template scripts with the Write tool.
- Simulators must not overflow a 375 px viewport (use `viewBox` SVGs, `min-w-0`, `flex-wrap`).
- Example programs must demonstrate the intended behaviour reliably (e.g. a training demo must actually converge
  with the seed you chose — check it).

### Trained-model simulators (pattern from round 5)

When a simulator needs a trained network: (1) commit a deterministic training script `web/scripts/train-<name>.py`
with a header documenting purpose, how to run and runtime; (2) export compact JSON (weights, config, vocab and a few
**verification vectors**) to `web/public/models/<name>.json`, ideally < ~300 KB; (3) write a self-contained TS
inference engine in `web/lib/sim/<name>.ts` and check in Node that it reproduces the PyTorch verification vectors
(~1e-4); (4) the simulator fetches the JSON with loading/error states and says in the UI that it is a real trained
model, on what data, and how big. Call `model.eval()` before exporting verification outputs. **Never download
pretrained weights** — train small models yourself on built-in or embedded public-domain data.

## 2. Simulators

A simulator is a default-exported React client component in `web/components/sims/<Name>Sim.tsx` (no props),
already registered in `web/components/sims/SimEmbed.tsx` and `registry.ts`.

- Use the shared primitives from `./SimShell`: `SimShell`, `useStepper`, `Controls`, `Slider`, `Narration`, `fmt`,
  `signedColor`, `seqColor`. Every simulator needs **Play, Pause, Step, Reset, Speed** (provided by `Controls`),
  interactive parameters, clear labels, live numeric values and a `Narration` panel explaining the current step.
- Put the pure math in `web/lib/sim/<name>.ts` (no React), so it can be unit-tested. Shared helpers:
  `web/lib/sim/random.ts` (seeded PRNG, matmul, softmax…).
- The visualization must be driven by the **real computation** — numbers shown must be the numbers computed.
- Rendering: SVG for diagrams/graphs (prefer), `<canvas>` for per-pixel/contour work. No animation libraries needed.
- Colours: only CSS variables / Tailwind tokens: `var(--accent)` (positive / forward), `var(--clay)` (negative /
  backward / gradients), `var(--gold)` (highlight / current), `var(--text)`, `var(--text-muted)`, `var(--border)`,
  `var(--surface)`, `var(--surface-2)`. Tailwind classes: `text-muted`, `bg-surface-2`, `border-border`, `text-accent`,
  `text-clay`, etc. Must look correct in both light and dark mode (`html[data-theme="dark"]`). No gradients, no glow.
- Responsive: SVGs use `viewBox` + `className="w-full h-auto"`; layout must work at 375 px width.
- Accessibility: buttons have labels, `aria-label` on SVG (`role="img"`), narration uses `aria-live` (Narration does).
- TypeScript strict mode must pass (`npx tsc --noEmit` in `web/`).

## 3. Plain-language standard (applies to every learner-facing sentence)

The site must teach like a patient friend sitting next to a complete beginner, not like a textbook. A learner with
zero background should finish any section thinking "oh, now I get it". Simple wording, **same technical depth**.

### Voice
- Talk to the reader: "you", "we", "let's". Friendly and encouraging, never condescending ("obviously", "trivially",
  "it is easy to see", "simply" are banned — if it were obvious we would not need to explain it).
- Short sentences (aim for ≤ 20 words; split anything longer). One idea per sentence. Short paragraphs (2–4 sentences).
- Everyday words: "use" not "utilize", "show" not "demonstrate", "because" not "owing to the fact that",
  "about" not "approximately", "start" not "commence", "so" not "consequently", "enough" not "sufficient".
- No academic filler or hedging stacks ("it should be noted that", "in the context of", "with respect to",
  "a non-trivial amount of"). No marketing tone.

### Explaining
- **What → why → how.** For every concept: what it is in one plain sentence, why we need it (the problem it solves),
  then how it works step by step.
- **Define before use.** The first time a technical term appears, explain it in plain words right there
  (e.g. "a **gradient** — the direction in which the loss goes up fastest"). Keep the real term (learners need it for
  interviews and papers), but never rely on a term before it has been explained in this lesson or linked.
- **Analogies from daily life** (cooking, sports, shopping, maps, school…), and say where the analogy stops working.
- **Break big ideas into small steps.** Prefer `steps` blocks and bullet lists over long paragraphs. Never stack
  several new concepts in one sentence; show how they relate.
- **Formulas:** keep every formula. Before or after each one, add a sentence that says what it means in words, and
  name each symbol in plain language ("$w$ is the weight — how much this input matters"). Worked numbers stay.
- **Hard parts get a simpler second pass:** if a paragraph is unavoidably dense, follow it with
  "In plain words: …" or a tiny concrete example.
- History, names and citations stay, but keep them short and explain why they matter to the learner.

### Length: simpler AND shorter
The user's rule: lessons must not get longer. A rewritten lesson's prose must be **at most as many words as the
original** (the checker fails above 105%); aim for 80–95%. Make room for plain-words explanations by cutting:
repetition (the same point made in intro, mechanism and summary), filler and hedging, long history asides,
tangents that don't help understanding, and sentences that restate a formula without adding meaning. Keep every
core idea, formula explanation, worked example, caveat that matters and practical tip — the goal is the best
understanding of the topic in the fewest clear words. Prefer one good analogy over three.

### What must NOT change (checked by `node scripts/check-prose-only.mjs`)
Code blocks (code, output, title, language), display formulas (`math` blocks and step `tex`), simulator embeds,
quiz answers / number of options / numeric answers and tolerances, exercise and interview-question counts, lesson
metadata (id, level, prerequisites, next, terms) and existing `/learn/` links. Rewrite prose only. Inline `$…$`
math inside prose may stay as is; do not change its meaning. Facts, numbers and claims must remain exactly as true
as before — simplifying never means dropping a caveat that matters or making something less accurate.
Quiz options keep their order (the `answer` index points into them).

### Example
Before: "The perceptron learning rule constitutes an error-driven update procedure wherein misclassified instances
induce an additive correction to the weight vector proportional to the input."

After: "The perceptron learns from its mistakes. Every time it gets an example wrong, it nudges its weights a little
toward the right answer. How big is the nudge? It is the input itself, scaled by the learning rate. Examples it
already gets right change nothing."
