# Code walkthrough — every file, in the order a document and a question travel through it

[HOW_IT_WAS_BUILT.md](HOW_IT_WAS_BUILT.md) explains the ideas and decisions. This page explains the
**code**. Keep the file open next to the section you are reading. Every section has:

* **Job** — why the file exists, in one sentence
* **How it works** — the steps in plain words
* **Lines worth knowing** — what an interviewer could point at
* **Say it in an interview** — one or two sentences to remember

The app has three tiers (MERN + Python): **client/** (React, in the browser) → **server/** (Node + Express,
owns MongoDB) → **ml-service/** (Python + FastAPI, owns the models and the search index). Parts A–F are the
Python ML code, Part G the Node server, Part H the React client.

```
UPLOAD:  client UploadZone → server routes/documents.js (multer saves file, Mongo doc "queued") → jobs.js
           → ML routers/index.py → engine.index_document → ingest/extract.py → ingest/chunk.py
                                    (search/embedder.py → search/indexes.py + chunk store)
           → jobs.js saves pages in MongoDB → ML /enrich → nlp/classify.py · nlp/ner.py · nlp/summarize.py
QUESTION: client AskPage → server routes/qa.js (chat session in MongoDB) → ML routers/qa.py
           → engine.ask → engine.search → engine.retrieve (indexes + fusion.py) → nlp/rerank.py
           → nlp/qa.py (extractive)  or  nlp/generate.py (LLM)   → server stores the answer → client
```

---

## Part A — Ingestion (`ml-service/ingest/`)

### A1. [`extract.py`](../ml-service/ingest/extract.py) — get clean text out of a file, page by page
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

### A2. [`chunk.py`](../ml-service/ingest/chunk.py) — cut pages into retrievable pieces
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

## Part B — Search (`ml-service/search/`)

### B1. [`embedder.py`](../ml-service/search/embedder.py) — text → vector
**Job:** wrap a Sentence-Transformers bi-encoder (BGE-small, 384 dimensions).
* `embed_passages()` — batches of chunks, `normalize_embeddings=True` so dot product = cosine similarity.
* `embed_query()` — adds BGE's query instruction ("Represent this sentence for searching relevant passages: ")
  — the model was trained that way; leaving it out costs accuracy.
* `count_tokens()` — used by the chunker.

### B2. [`indexes.py`](../ml-service/search/indexes.py) — two indexes
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

### B3. [`fusion.py`](../ml-service/search/fusion.py) — Reciprocal Rank Fusion
`score = Σ 1 / (60 + rank)` over the result lists. Uses ranks, not raw scores, because BM25 scores (0–30)
and cosine similarities (0–1) aren't comparable.

**Say it in an interview:** "I run BM25 and dense search in parallel and merge them with RRF, then re-rank
the top 30 with a cross-encoder. On my benchmark BM25 alone actually beat embeddings at rank 1 — SQuAD
questions reuse the paragraph's words — but hybrid had the best recall@20, which is what the re-ranker needs,
and the re-ranker gave the best rank-1 accuracy overall."

---

## Part C — NLP models (`ml-service/nlp/`)

### C1. [`rerank.py`](../ml-service/nlp/rerank.py)
`CrossEncoder.predict([(question, chunk), …])` — question and chunk are read **together**, so attention can
compare them word by word. More accurate than the bi-encoder, but one model call per pair → only on ~30.

### C2. [`qa.py`](../ml-service/nlp/qa.py) — extractive answers (the most interview-worthy file)
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

### C3. [`generate.py`](../ml-service/nlp/generate.py) — LLM answers with citations
Numbers the chunks `[1]…[5]` in the prompt, a system prompt says "only from these sources, cite them,
otherwise say 'I could not find this in the documents'". `apply_chat_template` formats it for Qwen;
greedy decoding (`do_sample=False`) keeps answers reproducible; a regex pulls the `[n]` citations back out
and ignores numbers that don't exist.

### C4. [`summarize.py`](../ml-service/nlp/summarize.py) — map-reduce summaries
DistilBART reads ≤ 1,024 tokens: split into 900-token sections → summarise each (map) → summarise the joined
summaries (reduce), repeat if still too long. Beam search (4 beams), `no_repeat_ngram_size=3` stops loops.

### C5. [`ner.py`](../ml-service/nlp/ner.py) and [`classify.py`](../ml-service/nlp/classify.py) — our fine-tuned models
NER: one BIO tag per token → use each word's **first** sub-token (as in training) → merge B-X, I-X… into
spans with character offsets → count entities across the document's chunks.
Classifier: softmax over 4 topics for the first 4 chunks, probabilities averaged.

### C6. [`common.py`](../ml-service/nlp/common.py) — `pick_device()`: GPU if available, else CPU.

---

## Part D — The ML web service (`ml-service/app/`)

This is a FastAPI app with **no database**. The Node server calls it; the browser never does.

### D1. [`config.py`](../ml-service/app/config.py)
Every setting with a default, overridable by `DI_*` environment variables (model names, chunk size,
`retrieve_k=30`, `rerank_k=5`, `enable_generator`, `data_dir`).

### D2. The chunk store (inside `engine.py`)
The ML service owns the search index the way a vector database would: `data/index/dense.faiss` (vectors) and
`data/index/chunks.json` (vector id → document id, file name, chunk text, pages, char offset). **The vector
id in FAISS is the key of the chunk store** — a search hit maps straight back to its text and page. Both files
are written with write-then-rename, so a crash never leaves half a file.

### D3. [`engine.py`](../ml-service/app/engine.py) — the heart
* `model(name)` — lazy loading with double-checked locking and **one lock per model**: each model is loaded
  once, on first use, and a slow load of one model never blocks another (a single global lock froze every
  question while the summariser was loading — see HOW_IT_WAS_BUILT §10). The 1.5 B LLM is never loaded unless
  someone asks a generative question.
* `load()` — at startup reads the chunk store, rebuilds BM25 from it and loads the FAISS file; if the counts
  disagree (file missing / crash mid-write) it re-embeds every chunk → vectors can always be rebuilt from text.
* `index_document()` — extract → chunk → embed (outside the lock: the slow part) → under one lock: drop the
  document's old chunks, assign new ids, add to FAISS + BM25, save. Indexing the same document twice replaces
  it, so the server can safely retry an interrupted job. `remove_document()` and `enrich()` (label, entities,
  summary) complete the set.
* `retrieve()` — BM25 and/or dense search (each asked for 2× the results, 4× when filtering by document),
  filter by document, RRF; also records each chunk's rank in each retriever (shown as "why" in the UI).
* `search()` — retrieve 30 → cross-encoder → keep 5, with timings.
* `ask()` — search → extractive span (with highlight offsets) **or** LLM answer with citations. Extractive
  has three outcomes by reader score: above the tuned margin → answer; between 0 and the margin → not found,
  plus a labelled low-confidence `suggestion`; ≤ 0 → nothing (HOW_IT_WAS_BUILT §10b).

### D4. Routers
* [`routers/index.py`](../ml-service/app/routers/index.py) — `POST /index/documents` (file path in, pages +
  chunk count out; 415 bad type, 404 missing file, 422 "no text found" for scans), `DELETE /index/documents/{id}`,
  `POST /enrich`. Plain `def` handlers, so FastAPI runs them in a thread and a long indexing job does not block
  questions.
* [`routers/qa.py`](../ml-service/app/routers/qa.py) — `/search` and `/ask`. The chat history comes **in the
  request** (the server keeps it in MongoDB), so this service stays stateless apart from the index. Model calls
  go through `run_in_threadpool` so the async server keeps serving other requests.

### D5. [`main.py`](../ml-service/app/main.py) and [`schemas.py`](../ml-service/app/schemas.py)
Startup: build the engine (load the embedder, load/rebuild the indexes). `/health`, `/metrics`. Pydantic schemas
validate input (e.g. `mode` must be hybrid/dense/bm25 → 422 otherwise) and document the API at `/docs`.
Document ids are strings — MongoDB ObjectIds created by the server.

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

## Part F — Python tests (`ml-service/tests/`)
`conftest.py` swaps in **fake models** (hash embedder, word-overlap re-ranker, sentence-picking "QA") so the
whole HTTP → extract → chunk → index → search → answer path is tested in seconds without downloads. Unit tests
cover extraction (incl. the three Unicode/hyphen regressions), chunk budgets/overlap/pages, RRF, FAISS
add/remove, BM25's formula. API tests cover index → search in 3 modes → document filter → exact answer
highlight → history → not-found → enrich → delete removes vectors → re-indexing replaces → the index survives a
restart → low-confidence suggestions → the generative gate → per-model locks → validation.

---

## Part G — The Node server (`server/`, Express + MongoDB)

The API the browser calls (`/api/...`). Its job is everything that is **not** model work: files, records,
background jobs, chat history. See also [server/README.md](../server/README.md).

| file | job |
|---|---|
| [`src/config.js`](../server/src/config.js) | settings from env vars / `.env`: port 4200, `MONGO_URL`, `ML_URL`, `DATA_DIR`, upload limit |
| [`src/db.js`](../server/src/db.js) | connect Mongoose; one JSON shape for every model (`id` string instead of `_id`, no `__v`) |
| [`src/models/Document.js`](../server/src/models/Document.js) | a document's status, enrich status, counts, label, entities, summary (`path` is never sent out) |
| [`src/models/Page.js`](../server/src/models/Page.js) | page texts, own collection + index `(document_id, number)` so the list stays light |
| [`src/models/ChatSession.js`](../server/src/models/ChatSession.js) | a chat with its messages **embedded** (always read together → embed, don't join) |
| [`src/ml.js`](../server/src/ml.js) | the only file that calls Python; forwards its status + message, 503 if unreachable; no 5-minute fetch timeout |
| [`src/jobs.js`](../server/src/jobs.js) | two in-memory queues (index, enrich), one job at a time each; `recover()` re-queues interrupted jobs at startup |
| [`src/routes/documents.js`](../server/src/routes/documents.js) | multer streams uploads to disk (type check → 415, size limit → 413), **202** + queue; delete: 409 while busy, removes vectors first |
| [`src/routes/qa.js`](../server/src/routes/qa.js) | `/search` (validate → ML), `/ask` (load/create session, send history, store both messages with sources), chats list/detail/delete |
| [`src/routes/system.js`](../server/src/routes/system.js) | `/health` = MongoDB status + ML status (always 200 so the UI can say which is down); `/metrics` |
| [`src/app.js`](../server/src/app.js) | builds the Express app (no `listen`, so tests use it directly); serves `client/dist` in production; every error → `{"detail": ...}` |
| [`src/index.js`](../server/src/index.js) | connect → recover jobs → listen |
| [`tests/api.test.js`](../server/tests/api.test.js) | real MongoDB + a **fake ML service** (tiny Express app): upload → ready → search → ask with history → delete; failures, slow enrichment, crash recovery, validation, health |

**Say it in an interview:** "Express is the system of record — it decides what exists (MongoDB) and when work
happens (job queues). Python only does maths on what it is given. If Python is down, the documents, pages and
chats are still there; only search and answering return 503."

## Part H — The React client (`client/`, Vite)

See [client/README.md](../client/README.md) for the file-by-file list. The essentials: `src/lib/api.js` is the
only file that calls `fetch` (one function per endpoint, errors turned into readable messages);
`src/App.jsx` maps URLs to pages with React Router; every page handles loading / empty / error states.

**Exercise:** upload a PDF, ask a question, and trace it from `client/src/pages/AskPage.jsx` → `server/src/routes/qa.js`
→ `ml-service/app/routers/qa.py::ask` → `engine.ask` → back to the highlighted span in the UI. If you can explain
every hop, you can answer any "how does it work" question.
