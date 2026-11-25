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

## How a question is answered

```
question ─┬─ BM25 keyword search ──┐
          └─ embedding search ─────┴─ RRF fusion → top 30 → cross-encoder re-rank → top 5 chunks
                                                                                     │
                     extractive: RoBERTa-SQuAD2 picks the best span (or "no answer") ┤
                     generative: Qwen2.5-1.5B writes an answer citing [1]..[5]  ─────┘ → answer + sources
```

Upload path: `file → extract text per page → clean → sentence-aware chunks (256 tokens, overlap) →
embed (BGE-small) → FAISS + BM25` in a background worker; then topic, entities and a map-reduce summary.

## Layout

```
doc-intelligence/
├── backend/
│   ├── ingest/     extract.py (PDF/DOCX/TXT → pages, cleaning) · chunk.py (sentence-aware chunking)
│   ├── search/     embedder.py · indexes.py (FAISS + BM25 written out) · fusion.py (RRF)
│   ├── nlp/        rerank.py · qa.py (extractive) · generate.py (LLM + citations) · summarize.py · ner.py · classify.py
│   ├── app/        main.py · config.py · db.py · engine.py (search + ask) · worker.py (ingestion) · routers/
│   └── tests/      pytest with tiny fake models (no downloads needed)
├── training/       download_assets.py · build_eval_corpus.py · finetune_ner.py · finetune_classifier.py
│                   eval_retrieval.py · eval_qa.py · eval_summarization.py · eval_generation.py · reports/
├── frontend/       Next.js UI (see frontend/README.md)
├── scripts/        dev_api.py · make_results.py
├── models/         our fine-tuned NER + classifier (generated)
└── docs/
```

## Run it

**Step-by-step local setup for both projects (one shared environment): see [../README.md](../README.md).**
Short version — create an environment:

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r training/requirements.txt -r backend/requirements-dev.txt
```

```bash
python training/download_assets.py          # models + datasets (~5 GB, once)
python training/finetune_ner.py             # ~5 min on a laptop GPU
python training/finetune_classifier.py      # ~6 min
python scripts/dev_api.py                   # API  http://localhost:8200/docs
cd frontend && npm install && npm run build && npm start   # UI http://localhost:3200
```

Reproduce the evaluation: `python training/build_eval_corpus.py`, then the four `eval_*.py` scripts,
then `python scripts/make_results.py` (writes docs/RESULTS.md). Tests: `cd backend && python -m pytest`.

Docker: `docker compose up --build` (CPU image; the LLM is disabled there by default — set
`DI_ENABLE_GENERATOR=true` on a GPU host). Settings: environment variables prefixed `DI_` (see `backend/app/config.py`).
