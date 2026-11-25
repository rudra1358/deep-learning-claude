# Document Intelligence — search and ask questions over your documents, with cited answers

Upload PDFs, Word files or text; then:

* **Ask** questions and get answers that **cite the exact source** (file + page), with the answer span
  highlighted — or "not found" when the documents don't contain it.
  * *Extractive* mode copies the answer from the document (cannot hallucinate).
  * *Generative* mode lets a local LLM (Qwen2.5-1.5B) write the answer, citing sources as [1], [2].
* **Search** semantically (meaning), by keyword (BM25) or both (hybrid), with cross-encoder re-ranking.
* Get every document **summarised**, **classified** by topic and its **people / organisations / places**
  extracted — the NER and topic models are fine-tuned in this repo.

Everything runs locally with open-weight models (CPU works; a 6 GB GPU makes it fast). Every quality
claim is measured on public labelled data — see [docs/RESULTS.md](docs/RESULTS.md).

**Preparing for an interview?** Read [docs/HOW_IT_WAS_BUILT.md](docs/HOW_IT_WAS_BUILT.md), then
[docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md), [docs/RESULTS.md](docs/RESULTS.md) and
[docs/INTERVIEW_QA.md](docs/INTERVIEW_QA.md).

## Architecture: MERN + a Python ML service

```
browser ── React (client/, :3200 in dev) ──/api──> Express (server/, :4200) ──HTTP──> FastAPI (ml-service/, :8200)
                                                       │                                 │
                                                    MongoDB                     models + FAISS + BM25
                                         (documents, pages, chat history)   (data/index: vectors + chunk store)
```

* **client/** — React 19 + Vite + plain JavaScript + Tailwind. The UI. Talks only to the Express server.
* **server/** — Node + Express + Mongoose. The system of record: uploads, documents, pages, chats, the
  background job queues. Calls the ML service for anything that needs a model.
* **ml-service/** — Python + FastAPI + PyTorch. All deep learning: extraction, chunking, embeddings, FAISS +
  BM25, re-ranking, the reader, the LLM, NER, topic classifier, summariser. Never called by the browser.

Why split it like this: [docs/HOW_IT_WAS_BUILT.md §8](docs/HOW_IT_WAS_BUILT.md#8-architecture-mern--a-python-ml-service).

## How a question is answered

```
question ─┬─ BM25 keyword search ──┐
          └─ embedding search ─────┴─ RRF fusion → top 30 → cross-encoder re-rank → top 5 chunks
                                                                                     │
                     extractive: RoBERTa-SQuAD2 picks the best span (or "no answer") ┤
                     generative: Qwen2.5-1.5B writes an answer citing [1]..[5]  ─────┘ → answer + sources
```

Upload path: Express saves the file and queues a job → the ML service does `extract text per page → clean →
sentence-aware chunks (256 tokens, overlap) → embed (BGE-small) → FAISS + BM25` → Express stores the pages in
MongoDB; then a second job gets topic, entities and a map-reduce summary.

## Layout

```
doc-intelligence/
├── client/         React UI (see client/README.md)
├── server/         Express + MongoDB API (see server/README.md)
├── ml-service/     Python ML service (FastAPI)
│   ├── ingest/     extract.py (PDF/DOCX/TXT → pages, cleaning) · chunk.py (sentence-aware chunking)
│   ├── search/     embedder.py · indexes.py (FAISS + BM25 written out) · fusion.py (RRF)
│   ├── nlp/        rerank.py · qa.py (extractive) · generate.py (LLM + citations) · summarize.py · ner.py · classify.py
│   ├── app/        main.py · config.py · engine.py (index + search + ask) · schemas.py · routers/ (index, qa)
│   └── tests/      pytest with tiny fake models (no downloads needed)
├── training/       download_assets.py · build_eval_corpus.py · finetune_ner.py · finetune_classifier.py
│                   eval_retrieval.py · eval_qa.py · eval_summarization.py · eval_generation.py · reports/
├── scripts/        dev_api.py (start the ML service) · make_results.py
├── models/         our fine-tuned NER + classifier (generated)
├── data/           uploads + the search index (generated, shared by server and ml-service)
└── docs/
```

## Run it

**Step-by-step local setup for both projects (one shared environment): see [../README.md](../README.md).**
Requirements: Python 3.12, Node 20+, MongoDB running on `mongodb://127.0.0.1:27017` (or set `MONGO_URL`).

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r training/requirements.txt -r ml-service/requirements-dev.txt
python training/download_assets.py          # models + datasets (~5 GB, once)
python training/finetune_ner.py             # ~5 min on a laptop GPU
python training/finetune_classifier.py      # ~6 min
```

Three terminals:

```bash
python scripts/dev_api.py                          # 1. ML service   http://localhost:8200/docs
cd server && npm install && npm start              # 2. API server   http://localhost:4200/api/health
cd client && npm install && npm run dev            # 3. UI (dev)     http://localhost:3200
```

Production-style: `cd client && npm run build`, then the server alone serves the UI at **http://localhost:4200**.

Tests: `cd ml-service && python -m pytest` (fake models) · `cd server && npm test` (real MongoDB, fake ML
service). Reproduce the evaluation: `python training/build_eval_corpus.py`, then the four `eval_*.py`
scripts, then `python scripts/make_results.py` (writes docs/RESULTS.md).

Docker: `docker compose up --build` → UI + API on http://localhost:4200 (CPU image; the LLM is disabled there
by default — set `DI_ENABLE_GENERATOR=true` on a GPU host). Settings: ML service `DI_*` variables (see
`ml-service/app/config.py`); server `PORT`, `MONGO_URL`, `ML_URL`, `DATA_DIR`, `MAX_UPLOAD_MB` (see
`server/.env.example`).
