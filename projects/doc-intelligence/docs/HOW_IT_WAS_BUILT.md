# How this project was built — explained simply

Read top to bottom once; afterwards use it as revision notes. Every number comes from
[RESULTS.md](RESULTS.md) (generated from measured reports). The code itself is explained in
[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md).

---

## 0. The 30-second pitch

> "I built a document question-answering system — retrieval-augmented generation done carefully. Documents
> are extracted page by page, cleaned, split into sentence-aware chunks and indexed twice: BM25 for keywords
> and BGE embeddings in FAISS for meaning. A question runs through both, the results are merged with
> reciprocal rank fusion and re-ranked by a cross-encoder. Then either an extractive reader copies the exact
> answer span — or says 'not found' — or a local 1.5B LLM writes an answer citing its sources. I also
> fine-tuned DistilBERT for entity extraction and topic classification. Everything is measured on public
> labelled data: re-ranking lifted rank-1 retrieval from 76 % to 85 %, and the 'no answer' threshold is tuned
> on one half of the questions and reported on the other."

Diagram to draw on the whiteboard:

```
UPLOAD   file → extract pages → clean → chunk (256 tok, overlap) → embed (BGE) → FAISS ┐
                                                                  → BM25 index          ┘   + topic, entities, summary
QUESTION question → BM25 top-60 ┐
                  → dense top-60┴→ RRF → top 30 → cross-encoder → top 5 → reader (span or "no answer")
                                                                        → or LLM (answer + [n] citations)
```

---

## 1. The problem

Companies have thousands of contracts, papers and reports. Keyword search misses paraphrases; a generic
chatbot invents answers. Users need **answers grounded in their own documents, with the exact source**, and
an honest "I don't know" when the documents don't say.

Requirements that follow from that:

| need | design decision |
|---|---|
| every answer checkable | chunks keep page + character offsets; the UI highlights the span |
| no invented answers | extractive mode copies text; generative mode must cite; "not found" is a valid answer |
| find paraphrases *and* exact terms | hybrid search: embeddings + BM25 |
| accurate top results | cross-encoder re-ranking of the shortlist |
| quality you can trust | labelled evaluation for every component |
| private / offline | all models open-weight and local |

---

## 2. Ingestion

**Extraction** — PyMuPDF reads the text layer of each PDF page; DOCX/TXT are grouped into pseudo-pages.
Cleaning turned out to matter more than expected (§10): Unicode normalisation (NFKC), soft hyphens, and
deciding when a line-end hyphen belongs to the word.

**Chunking** — why chunk at all? An embedding squeezes a text into *one* vector; a vector for 50 pages is a
blur. And models have input limits (BGE: 512 tokens). Design:
* **sentence-aware**: pack whole sentences, never cut one in half;
* **token budget** (256 of the embedder's own tokens), not characters;
* **overlap** (~40 tokens = the last sentence or two repeat at the start of the next chunk) so a fact at a
  boundary still appears with its context;
* every chunk remembers `page_start`, `page_end`, `char_start` → citations and highlighting.

**Chunk size** was chosen by experiment (128 / 256 / 384 tokens). Smaller chunks rank the right one first more
often (MRR 0.856 vs 0.848 vs 0.836), bigger chunks are easier to "hit" at depth 20 (98.7 / 99.4 / 99.5 %).
256 is the compromise: near-best deep recall for the re-ranker, half as many chunks as 128, and passages short
enough for the reader.

---

## 3. Retrieval

### Dense retrieval (bi-encoder)
BGE-small turns text into a 384-dim vector so that similar meaning → similar direction. Question and chunk
are encoded **separately**, so chunk vectors are computed once at upload and search is one matrix product.
Vectors are L2-normalised, so inner product = cosine similarity. FAISS `IndexFlatIP` does exact search
(milliseconds for 100k chunks; at millions you'd switch to an approximate index like HNSW or IVF).

### BM25 (keywords)
Score = Σ over query words of `idf × saturating term frequency × length normalisation`. Rare words count most,
repeats have diminishing returns (k1 = 1.2), long chunks are penalised (b = 0.75). Written by hand because the
library's textbook IDF goes negative for common words (§10).

### Why both? Hybrid with RRF
Embeddings understand paraphrases ("car" ≈ "vehicle") but are weak on exact rare tokens (names, codes,
numbers). BM25 is the opposite. **Reciprocal Rank Fusion** merges the two lists by rank —
`Σ 1/(60 + rank)` — because their raw scores live on different scales.

### Re-ranking (cross-encoder)
A cross-encoder reads question and chunk **together**, so attention compares them token by token — much more
accurate, but one transformer pass per pair, so only for the top 30. Retrieve-then-rerank = bi-encoder speed +
cross-encoder accuracy.

### What the measurements said (1,998 answerable questions, 929 chunks from 35 PDFs)

| method | Recall@1 | Recall@20 | MRR@10 | ms/query |
|---|---|---|---|---|
| BM25 | 80.8 | 97.9 | 0.867 | 0.8 |
| Dense MiniLM | 62.1 | 96.6 | 0.730 | 17.5 |
| Dense BGE | 65.0 | 97.0 | 0.754 | 20.6 |
| Hybrid (RRF) | 76.1 | **99.2** | 0.845 | 22.8 |
| Hybrid + re-rank | **85.1** | **99.4** | **0.903** | 124 |

Honest reading: **BM25 beat embeddings** here. SQuAD questions were written by people looking at the paragraph,
so they reuse its words — a known bias of this benchmark; on real user questions (paraphrased, vaguer) dense
retrieval usually does better. That is exactly why the system is hybrid: it is robust to both, gives the best
recall@20 (the pool the re-ranker sees) — and the re-ranker turns that into the best rank-1 accuracy.

---

## 4. Answering

### Extractive reader (default)
RoBERTa-base fine-tuned on SQuAD 2.0 predicts, for every token, a **start** score and an **end** score.
Best span = max of `start[i] + end[j]` (i ≤ j, ≤ 30 tokens, inside the passage). SQuAD 2.0 also trains it to
point at the `[CLS]` token when there is no answer: the **null score**. We answer only if
`best span − null > margin`. Long passages are read in overlapping windows (stride). Character offsets map the
span back into the chunk → exact highlight.

**Why not just use an LLM?** The extractive answer is *copied* from the document — it cannot be invented — and
it is cheap. Its limit: it can only answer with a short span.

**Tuning the "no answer" margin properly:** a higher margin means fewer wrong answers but more missed ones.
The margin is chosen on half of the evaluation questions and the result is reported on the *other* half —
tuning and reporting on the same data would overstate quality.

**Oracle vs system:** the reader is also evaluated with the gold paragraph handed to it ("oracle"). The gap
between oracle and full system shows how much retrieval costs. (Numbers: RESULTS §2.)

### Generative answers (RAG with citations)
Qwen2.5-1.5B-Instruct (fp16 on the 6 GB GPU) gets the top 5 chunks numbered `[1]…[5]` and a strict system
prompt: only these sources, cite every sentence, otherwise say exactly "I could not find this in the
documents." Greedy decoding for reproducibility. We parse the citations back to real chunks.

**What measuring it revealed (the most important lesson of the project):**

| variant (150 questions) | contains gold | cites a source | hallucinated on unanswerable | time |
|---|---|---|---|---|
| v1: instructions only | 74.7 % | 24.0 % | **86.7 %** | 2.6 s |
| v2: stricter rules + 2 worked examples | 78.7 % | 40.0 % | **97.3 %** (worse!) | 2.6 s |
| v2 + **reader gate** (default) | 72.0 % | 38.7 % | **24.0 %** | 1.6 s |

A small LLM ignores "say you don't know". Better prompting improved answers and citations but made it *more*
eager to answer everything. The fix was **architectural**: the extractive reader is good at "no answer"
(85 % correct), so it acts as an **answerability gate** — if it finds no answer in the sources, the LLM isn't
called. Hallucination 97 % → 24 %, 38 % faster, for 8 % of answerable questions refused. Lesson: measure
hallucination explicitly, and don't rely on the prompt alone for safety-critical behaviour.

### Conversation
Questions and answers are stored per chat session (with their sources). The generative mode gets the last
4 messages as context; extractive answers are per question.

---

## 5. Fine-tuning: NER and topic classification

Both start from **DistilBERT** (a 6-layer distilled BERT, 66 M parameters) — transfer learning: the encoder
already understands English; we train a small new head plus gently adjust the encoder.

* **NER (CoNLL-2003)** — token classification with BIO tags (B-PER, I-PER, …). Key detail: **label alignment**.
  "Washington" → "Wash", "##ington": the label goes on the first sub-token, the rest get −100 (ignored by the
  loss). Metric: entity-level F1 (seqeval) — an entity counts only if boundaries *and* type are right.
  Result: **F1 89.9** after 3 epochs (~5 minutes on a laptop GPU).
* **Topic classification (AG News)** — sequence classification from the [CLS] vector; 30k seeded articles
  (a quarter of the data, minutes instead of an hour). Result: **93.0 % accuracy**, macro-F1 93.0.

**The training loop** (written out in `training/common.py`, not hidden in a Trainer):
AdamW with small LR (2e-5–5e-5, big steps would erase pre-training) · weight decay except on biases/LayerNorm ·
linear warm-up for 10 % then linear decay (the new head starts random — warm-up avoids large noisy early
updates) · fp16 autocast + GradScaler (~2× faster, half memory) · gradient clipping at 1.0 · evaluate every epoch.

---

## 6. Summaries

DistilBART (CNN/DailyMail) reads at most 1,024 tokens, so long documents use **map-reduce**: summarise
900-token sections, then summarise the summaries. Evaluated with ROUGE against human highlights and compared
with LEAD-3 (just the first 3 sentences — a famously strong baseline on news). See RESULTS §6.

---

## 7. Evaluation design (the part that makes it "industry grade")

* **Real labels:** SQuAD 2.0 dev — 35 Wikipedia articles, 11,873 questions, ~50 % unanswerable.
* **Real path:** articles rendered to PDF and indexed together, exactly like user uploads.
* **Retrieval labels:** a chunk is relevant if it is from the right article and contains a gold answer.
* **Metrics:** Recall@k, MRR@10 (retrieval) · EM/F1 official SQuAD scripts logic (QA) · ROUGE (summaries) ·
  entity F1 (NER) · accuracy/macro-F1/confusion matrix (classification) · contains-gold / citation /
  refusal rates (LLM).
* **No tuning on the test numbers:** thresholds tuned on one split, reported on another; seeds fixed.
* **Know the benchmark's bias:** SQuAD favours keyword overlap — stated next to the numbers.

---

## 8. Backend design

* **FastAPI**; model work via `run_in_threadpool` so the event loop stays responsive.
* **Upload returns 202** immediately; a single background worker indexes (stage 1, searchable within seconds)
  and enriches (stage 2: topic, entities, summary). One worker = jobs never fight over the GPU. Scale-up: a
  queue (Celery/RQ + Redis) and more workers.
* **Crash recovery:** jobs live in memory, so on startup documents stuck in "processing" are re-queued.
* **Lazy model loading** with a lock: fast startup; the LLM only loads if someone uses it.
* **The DB is the source of truth:** chunk id = vector id; if the FAISS file is missing or out of sync, it is
  rebuilt from the database at startup.
* **SQLite** locally, **PostgreSQL** in Docker — same SQLAlchemy code.
* Validation: file types, size limit enforced *while streaming* the upload, query length, mode enums.

---

## 9. Testing

* Unit tests for the pure parts: cleaning rules, sentence splitting, chunk budgets/overlap/page tracking, RRF
  arithmetic, FAISS add/search/remove, BM25 formula and its regression case.
* API tests with **fake models** (hash embedder, word-overlap re-ranker, sentence-picking reader): upload →
  index → search in all modes → document filter → exact answer highlight → chat history → not-found →
  delete removes vectors → scanned PDF fails cleanly → crash recovery → slow enrichment never blocks indexing → a slow model load never blocks another → low-confidence suggestions → validation errors. 19 tests, ~8 s,
  no downloads — so they run in CI on every push.
* Model *quality* is covered by the evaluation scripts, not unit tests.

---

## 10. Real problems I hit (use these for "tell me about a challenge")

1. **BM25 found nothing for an obvious query.** A test searching "who walked on the Moon" in a Moon document
   got zero BM25 hits. *Cause:* the `rank_bm25` library's textbook IDF `log((N−n+0.5)/(n+0.5))` is negative
   when a word is in more than half the chunks (common in small collections). *Fix:* implemented BM25 myself
   with Lucene's `log(1 + …)`, plus a regression test.
2. **Half the answers vanished from the PDFs.** Measuring "does each gold answer still appear in the
   extracted text" first gave 98.7 % with the built-in PDF font, which can't encode "–", "−", "⊆". Switching to
   a Unicode font made it *48.8 %*: that font's spaces came out as non-breaking spaces (U+00A0), so phrases
   never matched. *Fix:* Unicode NFKC normalisation → 96.4 %.
3. **"Anglo-Saxons" became "AngloSaxons".** The PDF stored every hyphen as a *soft* hyphen (U+00AD), which my
   cleaner deleted. *Fix:* soft hyphen inside a line → "-", at a line end → decide using the document's own
   vocabulary (glue "exam-ple", keep "worst-case"). → **99.9 %** of answers survive extraction.
   *Lesson for all three:* measure the pipeline's intermediate outputs, not just the final score — these bugs
   would have quietly lowered every metric and every real user's results.
4. **A download hung at 0 bytes.** The Hugging Face hub's new transfer protocol stalled on this network;
   detected by watching file sizes, fixed with `HF_HUB_DISABLE_XET=1` (plain HTTPS).
5. **A dataset refused to load.** CoNLL-2003 on the Hub is a loading *script*, which `datasets` ≥ 4 no longer
   executes. The Hub keeps an auto-converted Parquet copy on `refs/convert/parquet` — same data, no code.
6. **Review found lifecycle gaps.** A document interrupted by a restart would stay "processing" forever, and a
   failed one showed enrichment "pending" forever. Added startup recovery and a "skipped" state, with tests.
   Live testing then found a gap in the recovery itself: documents *waiting* in the in-memory enrichment
   queue (state "pending", not yet "running") were not re-queued, so three documents never got summaries.
   Recovery now covers both states — lesson: a queue that lives in memory loses *everything* queued, not only
   the job that was running; that is why production systems use a persistent queue.
7. **One slow model froze every question.** In the live system `/api/ask` hung for minutes. *Cause:* lazy model
   loading used ONE global lock; the background worker held it while loading the summariser (whose weights
   were still downloading), so the question's re-ranker load waited behind it. *Fix:* one lock per model
   (still loads each model only once). Regression test: a slow load of model A must not delay model B.
8. **New uploads weren't searchable for minutes.** One worker did indexing *and* enrichment, so a slow summary
   blocked indexing of every later upload. *Fix:* two queues — index (fast, makes it searchable) and enrich
   (slow, optional). After the change, 4 documents were searchable within 5 s of a restart while enrichment
   was still waiting.
9. **The UI showed "confidence 1048 %".** The reader's score is a *logit margin* (span score − no-answer score),
   not a probability; the UI multiplied it by 100. Now shown as "reader score +10.5". Lesson: name units.

## 10b. Product decision: low-confidence suggestions

The tuned threshold (+2.75) maximises F1, but in live use it rejected correct answers: "Who was the Norse
leader?" → the reader found **Rollo** with score +1.2. And an unanswerable trick question ("Who did King Charles
III swear fealty to?" — it's the other way round) scored +1.5 with the wrong answer "Rollo". No threshold
separates those two. So the API now has three outcomes:

| reader score | response |
|---|---|
| > threshold | the answer, highlighted in its source |
| 0 … threshold | "Not found in your documents" **plus** "Possible answer (low confidence): …", highlighted, with an explanation |
| ≤ 0 (model prefers "no answer") | not found, nothing suggested |

The evaluation still counts only real answers; the suggestion just keeps a missed-but-correct answer one
click away without claiming it. This is the difference between optimising a metric and designing for users.

## 10c. Error analysis: a confident wrong answer

"Who patented the first steam engine?" → **"Thomas Newcomen"**, score +5.6 (above threshold) — wrong; the
document says Jerónimo de Ayanz y Beaumont obtained the first steam-engine patent in 1606. Diagnosis:
retrieval was fine (the right chunk was rank 2); the **reader** preferred "invented by Thomas Newcomen" from a
sentence about the *first commercially successful* engine — it treated "invented" ≈ "patented". Root cause
in the design: the reader scores each passage **independently**, so scores are not calibrated across
passages. Fixes to try: score answers jointly over all passages, use a stronger reader (RoBERTa-large /
DeBERTa), or let the generative mode compare sources. A good reminder that ~78 F1 means about one in five
answers is imperfect — which is why every answer shows its source.

## 10d. Known model limitation: topic labels

The topic classifier was trained on AG News (World / Sports / Business / Sci/Tech). A history article about the
Normans got "Sci/Tech" with only 49 % confidence — there is no "History" class. The label set must match the
customer's needs; low-confidence labels should be shown as "uncertain".

---

## 11. Limitations and next steps

* **Benchmark bias:** SQuAD favours BM25; test on real user questions (or BEIR datasets) before claiming
  dense > sparse or vice versa.
* **Scanned PDFs** need OCR (e.g. Tesseract / PaddleOCR); tables and multi-column layouts need layout-aware
  extraction.
* **Scale:** exact FAISS + in-memory BM25 are fine to ~10⁵–10⁶ chunks; beyond that an ANN index (HNSW) and a
  search engine (OpenSearch) or pgvector + full-text search.
* **Generative faithfulness** is measured coarsely (contains-gold, citations, refusals); a stricter check would
  verify each cited sentence entails the claim (an NLI model or an LLM judge).
* **Chat follow-ups** ("and when did he die?") are retrieved without rewriting the question using history;
  query rewriting would fix pronoun questions.
* **Security:** no authentication or per-user document isolation yet; uploads are served back as-is.
