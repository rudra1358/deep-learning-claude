# Development Roadmap

Phases follow the master specification. ✅ done · 🟡 partly done · ⬜ not started.

| Phase | Scope | State |
|---|---|---|
| 1 Research & planning | Curriculum hierarchy (21 modules, ~85 topics, gaps added: NumPy module, linear/logistic regression bridge, LLM module), architecture, content contract | ✅ |
| 2 Foundation | Next.js app, design tokens (light/dark), layout, routing, navigation, responsive shell | ✅ |
| 3 Platform | Home, curriculum, roadmap, lesson pages (sidebar / TOC / prev-next / breadcrumbs), KaTeX, code blocks, glossary, search, progress, bookmarks, quizzes | ✅ |
| 4 Visualization engine | Shared simulator framework + 39 simulators (6 backed by real trained models) — every simulator in the spec exists | ✅ (browser QA ongoing) |
| 5 Curriculum content | 63 full lessons; Modules 0–14 complete (see STATUS.md); ~23 topics remain (Modules 15–20) | 🟡 |
| 6 Code playground & assessments | Pyodide playground, per-lesson quizzes & exercises, practice hub | 🟡 (Monaco editor, isolated server-side execution not yet) |
| 7 Industry projects | Designs, milestones, repo structures published; implementations not started | 🟡 |
| 8 QA | Type checks, build, backend tests, lesson code executed; full browser QA of every simulator | 🟡 |
| 9 Optimization | Static generation, code-split simulators; Lighthouse/a11y audit pending | 🟡 |

## Next steps, in priority order (the user's suggested order)

1. ~~Finish Modules 2–4~~ ✅ done (round 2).
2. ~~Modules 0–1~~ ✅ done (round 7, incl. the broadcasting simulator).
   ~~Modules 5–7~~ ✅ done (round 3, incl. batch-norm and dropout simulators).
3. ~~Modules 8–9~~ ✅ done (round 4, incl. RNN, BPTT and GRU simulators).
5. ~~Modules 10–11~~ ✅ done (round 5, incl. the trained-model Transformer flagship simulator).
6. ~~Modules 12–13~~ ✅ done (round 6, incl. detection and word-embeddings simulators).
7. ~~Module 14~~ ✅ done (round 6, incl. trained autoencoder, VAE and diffusion simulators and a live-training GAN).
8. **Modules 15–20**.
9. **Projects (narrowed to two, 2026-11-24)**: Project 1 Vision Analytics (`projects/vision-analytics`, in progress)
   and Project 2 Context-Aware Document Intelligence (RAG). The diffusion studio (Project 3) is dropped.
   Both projects must keep their code simple and ship a file-by-file `docs/CODE_WALKTHROUGH.md`.

## Spec simulators
All built (round 6 completed the list). Self-attention and multi-head views live inside the attention explorer and
the trained Transformer explorer rather than as separate pages.
