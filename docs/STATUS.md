# Implementation Status

_Last updated: 2026-11-25 (round 7: Modules 0–1 completed — the curriculum is now complete from Module 0 through Module 14)._

## Complete and verified

**Modules 0–14 are fully written** (every topic has a full 16-section lesson): 63 lessons in total.

**Platform** (`web/`, Next.js production build: 144 static pages, `tsc --noEmit` clean)
- Pages: home, curriculum, roadmap, lesson pages, playground (+40 simulator pages), neural network visualizer,
  math lab, architecture explorer, code playground, practice hub, projects, research papers, glossary (481 terms, all 223 formulas KaTeX-checked),
  resources, progress dashboard, 404.
- Light and dark themes; **all 117 key pages (every lesson, simulator and main page) measured at 375 px with no horizontal overflow** (long display and inline
  formulas scroll inside their own box; long inline code and overlong plain-text identifiers wrap; tab bars scroll).
- Search (Ctrl K or /) over modules, topics, every lesson section, code, formulas, glossary, simulators, papers.
- Progress, quizzes, bookmarks, project milestones, streaks (localStorage + optional server sync).
- Crawl: 140 internal pages, 0 broken links.

**Simulators (40)** — all render, step and play with no runtime errors or NaN (automated check):
network, convolution, attention (flagships), cnn-pipeline, neuron, activations, optimizers, gradient-descent,
loss-functions, fitting, matmul, chain-rule, positional-encoding, lstm, tokenizer, **linear-transform (eigen/SVD),
distributions (sampling + MLE/MAP), entropy (H / CE / KL), logistic-regression, approximator (universal
approximation), gradient-flow (vanishing/exploding), batchnorm, dropout, lr-schedule, regularization (L1/L2 geometry), metric-learning (contrastive/triplet), receptive-field, conv-variants (dilated/transposed/separable), rnn, bptt, gru**, and two backed by **real trained models running in the browser**: **transformer** (flagship: 61,488-param char-level Transformer trained on Alice ch. 1; browser logits match PyTorch to 5.4e-6 — re-verified independently) and **seq2seq-attention** (Bahdanau-attention GRU on string reversal; 100% at all trained lengths vs 0% for the fixed-context baseline at length ≥ 16). Training scripts: `web/scripts/train-char-transformer.py`, `web/scripts/train-seq2seq-attention.py`; weights in `web/public/models/`. The seq2seq script was re-run end to end and reproduced the shipped weights bit-for-bit (all 30,509 values identical) and its accuracy table.

Round 6 added **detection** (IoU / anchors / NMS, live) and five more real-model simulators, each engine independently re-checked against its PyTorch/Python reference in Node:
- **autoencoder** and **vae** (trained on sklearn digits; max |Δ| 5.5e-6 over 1,932 values and 6.5e-7 over 978) — `web/scripts/train-autoencoders.py`
- **diffusion** (DDPM/DDIM on two-moons + class-conditional 8×8 digits with classifier-free guidance; max |Δ| 1.4e-11 over 428 values) — `web/scripts/train-diffusion.py`
- **word-embeddings** (skip-gram, 1,714 words × 50 d; 60 stored results reproduce, max cosine Δ 4.9e-7). **Caveat:** trained on a script-generated "Toyland English" corpus (no public-domain books were available offline and no download was approved), so its 62.2% analogy accuracy is an upper bound for real text — `web/scripts/train-word2vec.py`
- **gan** — trains live in the browser (hand-written backprop + Adam, gradient-checked). Re-run under Node: the "Converges" preset reaches 8/8 ring modes by iteration 5,000 (82.5% high-quality at 8,000); the "D too strong (non-saturating)" preset collapses to 1 mode (100% of samples), exactly as documented; ≈2.8 ms/iteration.

**Full lessons (63)** — independently verified by `npm run check:lessons` (see below):
- m00 (all 3): numpy-arrays, vectorization, plotting-data — plus the **broadcasting** simulator (logic matches NumPy, incl. its error message, on 400 random shape pairs)
- m01 (all 3): ai-ml-dl, history, dl-workflow (history claims reviewed: the 16 the author flagged as uncertain were checked against the record)
- m02 (all 9): vectors-matrices, norms-spaces, eigen-svd, derivatives-chain-rule, jacobians-hessians, probability,
  mle-map, information-theory, optimization-math
- m03 (all 5): linear-logistic-regression, artificial-neuron, activation-functions, multilayer-networks, training-basics
- m04 (all 4): forward-propagation, backpropagation, gradient-problems, gradient-checking
- m05 (all 3): regression-losses, classification-losses, advanced-losses
- m06 (all 3): gradient-descent, momentum-adaptive, lr-schedules
- m07 (all 4): initialization, normalization, regularization, residual-connections
- m08 (all 5): images-as-tensors, convolution, pooling-receptive-field, conv-variants, cnn-architectures
- m09 (all 4): rnn-basics, bptt, lstm, gru-variants
- m10 (all 4): attention-intro, scaled-dot-product-attention, multi-head-attention, positional-encoding
- m11 (all 4): transformer-architecture, tokens-embeddings, training-inference, transformer-families
- m12 (all 4): transfer-learning, object-detection, segmentation, vision-tasks
- m13 (all 4): text-preprocessing, word-embeddings, nlp-tasks, language-modeling
- m14 (all 4): autoencoders, vae, gans, diffusion

**Lesson checker** (`web/scripts/check-lessons.mjs`, `npm run check:lessons`): for every lesson verifies all 16
sections, interview (≥3/3/2) / quiz (≥6) / exercise (≥2) counts, a simulator embed, strict KaTeX rendering of every
formula (16,099), every internal link, and **re-executes every Python block, requiring byte-identical output**
(263 programs across 63 lessons; all pass). Blocks titled “not executed in CI” (TensorFlow) or “vary between runs” are skipped.

**Backend** (`backend/`): FastAPI progress API, 5/5 pytest tests passing.

## Known limitations / to review
- TensorFlow/Keras tabs are not executed (TensorFlow not installable on Python 3.14); labelled as such.
- Citations written from authors' memory, not re-checked against sources — mainstream but deserve an editorial pass:
  e.g. Greff et al. 2017, Jozefowicz et al. 2015, Pearlmutter 1994, Martens & Grosse 2015, Dauphin et al. 2014,
  Leshno et al. 1993, Barron 1993, Montúfar et al. 2014, Telgarsky 2016, BLOOM/ALiBi, LLaMA/RoPE, YOLOv4/Mish.
- Lesson HTML is large (0.5–2.3 MB uncompressed) because KaTeX is pre-rendered; consider trimming the RSC payload.
- Some lessons' “Run in browser” examples use scikit-learn datasets (`load_digits`, `load_sample_image`) that have not
  been tested under Pyodide.
- Most new simulators were checked programmatically plus a visual spot-check; not every control was exercised by hand.
- Several lessons' PyTorch examples take 30–90 s when other jobs load the CPU (mostly torch import time); the
  checker's per-block timeout was raised to 240 s accordingly.
- The word-embeddings model uses a generated corpus (see above); retraining on real public-domain text would need a download.

## Not started
~23 remaining topic lessons (Modules 15–20), project implementations — see ROADMAP.md.
All simulators named in the original specification now exist.
