# Brief: plain-language rewrite of lessons (round 9)

You are rewriting lessons of "Deep Learning University" (Next.js curriculum, web app in `web/`) into simple,
beginner-friendly English that is **also shorter**. The user's words: explanations were too academic; make every
lesson feel like a patient, friendly teacher explaining to a complete beginner — "not that short, but the main thing
is the user gets the best knowledge of the topic". Lessons must not get longer.

## Read first
- `docs/CONTENT_GUIDE.md` section 3 "Plain-language standard" (voice, explaining, **length budget**, what must not
  change) and section 1 (inline dialect: `**bold**`, `` `code` ``, `$LaTeX$`, `[text](/learn/...)`; use `String.raw`
  for strings containing backslashes, as the files already do).
- `web/lib/types.ts` (Block types).
- Model of the target tone: `web/content/lessons/m03-artificial-neuron.ts` and
  `m10-scaled-dot-product-attention.ts` are the pilots (tone approved by the user; they are being trimmed to the same
  length budget, so copy their tone, not their length).

## What to rewrite
Every learner-facing prose string: `summary`, and in all 16 sections every `p`, `h`, `list` item, `callout`
(title + text), `table` cell, `steps` title/text, `math`/`sim` caption, `details` summary, interview `qa` question and
answer, quiz question / options / explanation, exercise title / prompt / hint, and prose inside exercise solutions.
You may merge, split, reorder or delete prose blocks, turn paragraphs into lists/steps, and add short analogies or
"In plain words" lines — as long as the total stays within budget.

Priorities, in order: (1) correct and complete core knowledge, (2) a beginner can follow it, (3) short.
Cut repetition, filler, hedging, long history asides and tangents to pay for clearer explanations.

## Must NOT change (enforced)
Code blocks (code, output, title, lang), `math` block tex, step `tex`, `sim` ids and order, quiz answers / option
order / option count / numeric answers / tolerances, exercise count, interview questions per level (may not drop),
lesson metadata (id, level, prerequisites, next, terms) and existing `/learn/...` links. Facts and numbers must stay
exactly as true as before; keep caveats that matter.

## Verify each lesson (run from `web/`; both must pass)
1. `node scripts/check-prose-only.mjs <file-substring>` → `ok` (diffs against git HEAD: protected content identical,
   prose words ≤ 105% of original — aim for 80–95%).
2. `node scripts/check-lessons.mjs <file-substring> --no-python` → `ok` (sections, counts, strict KaTeX, links).
   Python is skipped because check 1 proves code and outputs are byte-identical.
Also grep your files for banned words: simply, obviously, trivially, utilize, "easy to see", "it should be noted".

## Rules
- Edit only your assigned lesson files. Do not run `npm run build`, `next build` or `next dev`. Do not commit.
- Work one lesson at a time; finish and verify it before starting the next.
- Final report (short): one checker line per lesson (with its word %), anything you were unsure about.
