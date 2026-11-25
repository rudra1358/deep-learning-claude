# Code walkthrough — every file, in the order a document and a question travel through it

[HOW_IT_WAS_BUILT.md](HOW_IT_WAS_BUILT.md) explains the ideas and decisions. This page explains the
**code**. Keep the file open next to the section you are reading. Every section has:

* **Job** — why the file exists, in one sentence
* **How it works** — the steps in plain words
* **Lines worth knowing** — what an interviewer could point at
* **Say it in an interview** — one or two sentences to remember

```
UPLOAD:  routers/documents.py → worker.py → ingest/extract.py → ingest/chunk.py → engine.add_chunks
                                    │                                   (search/embedder.py → search/indexes.py)
                                    └→ nlp/classify.py · nlp/ner.py · nlp/summarize.py   (enrichment)
QUESTION: routers/qa.py → engine.ask → engine.search → engine.retrieve (indexes + fusion.py)
                                                      → nlp/rerank.py
                                     → nlp/qa.py (extractive)  or  nlp/generate.py (LLM)
```

---

## Part A — Ingestion (`backend/ingest/`)

### A1. [`extract.py`](../backend/ingest/extract.py) — get clean text out of a file, page by page
**Job:** turn PDF / DOCX / TXT into a list of `Page(number, text)`, because every answer must cite a page.

**How it works**
1. PDF → PyMuPDF `page.get_text("text")` for each page. DOCX/TXT have no real pages, so paragraphs are
   grouped into ~3,000-character "pages" (`_split_into_pages`) to keep citations useful.
2. `vocabulary()` collects every word of the document; `clean()` then fixes PDF artefacts:
   * `unicodedata.normalize("NFKC", …)` — non-breaking spaces become normal spaces, "ﬁ" becomes "fi";
   * soft hyphens (U+00AD) become real hyphens (some PDF generators store every "-" that way);
   * a hyphen at a line end is removed **only if** the glued word exists elsewhere in the document
     ("exam-/ple" → "example") — otherwise it is kept ("Anglo-/Saxons" → "Anglo-Saxons");
   * line breaks inside a paragraph become spaces.
3. A PDF with no text layer (a scan) raises "no text found" instead of silently indexing nothing.

**Lines worth knowing:** `return a + b if (a + b).lower() in vocab else f"{a}-{b}"` — letting the document
decide how to repair hyphens. Each of the three cleaning rules was added because the evaluation caught a
real failure (see HOW_IT_WAS_BUILT §10); together they took answer survival from 48.8 % to 99.9 %.

**Say it in an interview:** "Text extraction looks trivial but it silently breaks retrieval: one PDF font
turned every space into a non-breaking space and half my answers became unfindable. I measure how many gold
answers survive extraction, and normalise Unicode and hyphens based on the document's own vocabulary."

### A2. [`chunk.py`](../backend/ingest/chunk.py) — cut pages into retrievable pieces
**Job:** produce `Chunk(index, text, page_start, page_end, char_start)` objects of at most N tokens.

**How it works**
1. `split_sentences()` splits on `. ! ?` followed by a capital letter, keeping each sentence's offset.
2. `chunk_pages()` walks through all sentences (remembering their page) and packs **whole sentences** into
   a chunk until adding the next would exceed `max_tokens` (256).
3. It then steps back over the last ~40 tokens of sentences so the next chunk starts with them (overlap).
4. A single sentence longer than the budget still becomes its own chunk (never dropped).

**Lines worth knowing:** `while k - 1 > i and back + sents[k - 1][3] <= overlap_tokens:` — the overlap step;
`k - 1 > i` guarantees progress, so the loop can never get stuck.
`count_tokens` is passed in, so chunk sizes are in the embedding model's own tokens.

**Say it in an interview:** "Chunks are sentence-aware with token budgets and overlap, and each chunk keeps
its page and character offset so the UI can highlight the exact source."

---

## Part B — Search (`backend/search/`)

### B1. [`embedder.py`](../backend/search/embedder.py) — text → vector
**Job:** wrap a Sentence-Transformers bi-encoder (BGE-small, 384 dimensions).
* `embed_passages()` — batches of chunks, `normalize_embeddings=True` so dot product = cosine similarity.
* `embed_query()` — adds BGE's query instruction ("Represent this sentence for searching relevant passages: ")
  — the model was trained that way; leaving it out costs accuracy.
* `count_tokens()` — used by the chunker.

### B2. [`indexes.py`](../backend/search/indexes.py) — two indexes
* `DenseIndex` — FAISS `IndexIDMap2(IndexFlatIP)`: exact inner-product search; our database chunk ids are
  used as FAISS ids so results map straight back and a deleted document's vectors can be removed. A lock
  protects writes.
* `BM25Index` — **BM25 written out by hand** (~25 lines). `set_all()` counts words per chunk and computes
  `idf = log(1 + (N - n + 0.5) / (n + 0.5))`; `search()` adds up `idf * f*(k1+1) / (f + k1*(1-b+b*len/avg))`
  for each query word. `tokenize()` lower-cases and removes stop-words.

**Lines worth knowing:** the `1 +` inside the log (Lucene's IDF). The `rank_bm25` library uses the textbook
formula, which turns **negative** for words in more than half the chunks — a test caught a "Moon" query
returning nothing in a Moon document, so BM25 was reimplemented (regression test:
`test_bm25_common_word_still_scores_positive`).

### B3. [`fusion.py`](../backend/search/fusion.py) — Reciprocal Rank Fusion
`score = Σ 1 / (60 + rank)` over the result lists. Uses ranks, not raw scores, because BM25 scores (0–30)
and cosine similarities (0–1) aren't comparable.

**Say it in an interview:** "I run BM25 and dense search in parallel and merge them with RRF, then re-rank
the top 30 with a cross-encoder. On my benchmark BM25 alone actually beat embeddings at rank 1 — SQuAD
questions reuse the paragraph's words — but hybrid had the best recall@20, which is what the re-ranker needs,
and the re-ranker gave the best rank-1 accuracy overall."

---

## Part C — NLP models (`backend/nlp/`)

### C1. [`rerank.py`](../backend/nlp/rerank.py)
`CrossEncoder.predict([(question, chunk), …])` — question and chunk are read **together**, so attention can
compare them word by word. More accurate than the bi-encoder, but one model call per pair → only on ~30.

### C2. [`qa.py`](../backend/nlp/qa.py) — extractive answers (the most interview-worthy file)
1. Tokenise `(question, passage)` pairs with `truncation="only_second"`, `stride=128`,
   `return_overflowing_tokens=True` → long passages become overlapping 384-token windows;
   `return_offsets_mapping=True` → every token knows its character position.
2. The model returns a start score and an end score per token.
3. For each window: only passage tokens are allowed (`sequence_ids == 1`), take the top 20 starts × top 20
   ends, keep the best valid pair (`end >= start`, length ≤ 30 tokens).
4. `score = best span − null score` where null = `start[CLS] + end[CLS]` (SQuAD 2.0's "no answer").
5. Offsets map the span back to exact characters → `Span(text, score, passage, char_start, char_end)`.
6. `answer()` returns the best span over all passages, or `None` if its score ≤ the tuned margin.

**Say it in an interview:** "The reader predicts start and end positions; I compare the best span with the
model's 'no answer' score and only answer above a threshold that I tuned on held-out questions."

### C3. [`generate.py`](../backend/nlp/generate.py) — LLM answers with citations
Numbers the chunks `[1]…[5]` in the prompt, a system prompt says "only from these sources, cite them,
otherwise say 'I could not find this in the documents'". `apply_chat_template` formats it for Qwen;
greedy decoding (`do_sample=False`) keeps answers reproducible; a regex pulls the `[n]` citations back out
and ignores numbers that don't exist.

### C4. [`summarize.py`](../backend/nlp/summarize.py) — map-reduce summaries
DistilBART reads ≤ 1,024 tokens: split into 900-token sections → summarise each (map) → summarise the joined
summaries (reduce), repeat if still too long. Beam search (4 beams), `no_repeat_ngram_size=3` stops loops.

### C5. [`ner.py`](../backend/nlp/ner.py) and [`classify.py`](../backend/nlp/classify.py) — our fine-tuned models
NER: one BIO tag per token → use each word's **first** sub-token (as in training) → merge B-X, I-X… into
spans with character offsets → count entities across the document's chunks.
Classifier: softmax over 4 topics for the first 4 chunks, probabilities averaged.

### C6. [`common.py`](../backend/nlp/common.py) — `pick_device()`: GPU if available, else CPU.

---

## Part D — The web service (`backend/app/`)

### D1. [`config.py`](../backend/app/config.py)
Every setting with a default, overridable by `DI_*` environment variables (model names, chunk size,
`retrieve_k=30`, `rerank_k=5`, upload limit, `enable_generator`).

### D2. [`db.py`](../backend/app/db.py)
Tables `documents`, `pages`, `chunks`, `chat_sessions`, `chat_messages`. **A chunk's primary key is also its
vector id in FAISS** — the link between the database and the index. SQLite gets `foreign_keys=ON` (cascade
deletes) and WAL mode (readers don't block the worker's writes).

### D3. [`engine.py`](../backend/app/engine.py) — the heart
* `model(name)` — lazy loading with double-checked locking and **one lock per model**: each model is loaded
  once, on first use, and a slow load of one model never blocks another (a single global lock froze every
  question while the summariser was loading — see HOW_IT_WAS_BUILT §10). The 1.5 B LLM is never loaded unless
  someone asks a generative question.
* `load()` — at startup rebuilds BM25 from the DB and loads the FAISS file; if the counts disagree (file
  missing / crash mid-write) it re-embeds everything → the index can always be rebuilt from the database.
* `retrieve()` — BM25 and/or dense search (each asked for 2× the results, 4× when filtering by document),
  filter by document, RRF; also records each chunk's rank in each retriever (shown as "why" in the UI).
* `search()` — retrieve 30 → cross-encoder → keep 5, with timings.
* `ask()` — search → extractive span (with highlight offsets) **or** LLM answer with citations. Extractive
  has three outcomes by reader score: above the tuned margin → answer; between 0 and the margin → not found,
  plus a labelled low-confidence `suggestion`; ≤ 0 → nothing (HOW_IT_WAS_BUILT §10b).

### D4. [`worker.py`](../backend/app/worker.py) — background ingestion
**Two** one-thread queues. Index queue (`_run_index` → `_index`): extract → chunk → save → commit (assigns
ids) → `engine.add_chunks` → `ready`, then hands the document to the enrich queue (`_run_enrich` →
`_enrich`): label, entities, summary. Separate queues so a slow summary never delays the next upload becoming
searchable. Failures are recorded on the row and never crash a worker. `recover()` re-queues interrupted jobs
to the right queue at startup.

### D5. Routers
* [`routers/documents.py`](../backend/app/routers/documents.py) — upload streams files to disk in 1 MB chunks
  (size limit enforced while streaming), returns **202** and queues the job; delete removes vectors first,
  then rows and file; 409 while a document is still processing.
* [`routers/qa.py`](../backend/app/routers/qa.py) — `/search`, `/ask` (creates or continues a chat session,
  passes history, stores the answer *with its sources*), chat list/detail/delete. Model calls go through
  `run_in_threadpool` so the async server keeps serving other requests.

### D6. [`main.py`](../backend/app/main.py) and [`schemas.py`](../backend/app/schemas.py)
Startup: create tables → build engine/indexes → recover interrupted jobs. `/api/health`, `/api/metrics`.
Pydantic schemas validate input (e.g. `mode` must be hybrid/dense/bm25 → 422 otherwise) and document the API.

---

## Part E — Training and evaluation (`training/`)

| file | does |
|---|---|
| `common.py` | paths, seeding, `update_metrics()`, and **the training loop**: AdamW, warm-up + linear decay, fp16 autocast + GradScaler, gradient clipping, evaluation after each epoch |
| `download_assets.py` | downloads every model/dataset once (CoNLL comes from the Hub's Parquet branch) |
| `finetune_ner.py` | CoNLL-2003; label alignment (first sub-token gets the label, others −100); seqeval entity F1 |
| `finetune_classifier.py` | AG News, 30k-article seeded subset; accuracy, macro-F1, confusion matrix |
| `build_eval_corpus.py` | SQuAD 2.0 dev → 35 PDFs + 11,873 labelled questions |
| `eval_retrieval.py` | chunk-size sweep and BM25 / dense / hybrid / re-rank comparison (Recall@k, MRR) |
| `eval_qa.py` | official SQuAD EM/F1; oracle vs system; margin tuned on half, reported on the other half |
| `eval_summarization.py` | ROUGE vs the LEAD-3 baseline |
| `eval_generation.py` | LLM: contains-gold, citation rate, refusal on unanswerable questions |

## Part F — Tests (`backend/tests/`)
`conftest.py` swaps in **fake models** (hash embedder, word-overlap re-ranker, sentence-picking "QA") so the
whole HTTP → worker → index → search → DB path is tested in seconds without downloads. Unit tests cover
extraction (incl. the three Unicode/hyphen regressions), chunk budgets/overlap/pages, RRF, FAISS add/remove,
BM25's formula. API tests cover upload → ready → search in 3 modes → document filter → exact answer
highlight → chat history → not-found → delete removes vectors → failed scans → crash recovery → validation.

**Exercise:** upload a PDF, ask a question, and trace it from `routers/qa.py::ask` to the highlighted span in
the UI. If you can explain every hop, you can answer any "how does it work" question.
