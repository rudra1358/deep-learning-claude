"""Measure retrieval quality on the SQuAD 2.0 corpus (35 articles as PDFs, one shared index).

Uses the SAME backend code as the API (backend/ingest, backend/search, backend/nlp/rerank).

Labels: a chunk is relevant to a question if it comes from the question's article AND contains
one of the gold answer strings. Only answerable questions are used here (a seeded sample).

Metrics
  Recall@k  - share of questions with at least one relevant chunk in the top k
              (if it is not retrieved, the answerer can never find it)
  MRR@10    - mean of 1 / rank of the first relevant chunk (0 if not in the top 10);
              rewards putting the right chunk FIRST, which matters because we only pass ~5 on
Experiments
  1. chunk size sweep (128 / 256 / 384 tokens) with hybrid retrieval
  2. retriever comparison at the chosen size: BM25, dense (MiniLM), dense (BGE), hybrid (BM25 + BGE),
     hybrid + cross-encoder re-ranking

    python training/eval_retrieval.py [--questions 2000]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

import numpy as np

from common import DATA, ROOT, update_metrics

sys.path.insert(0, str(ROOT / "backend"))
from ingest.chunk import chunk_pages  # noqa: E402
from ingest.extract import extract  # noqa: E402
from nlp.rerank import Reranker  # noqa: E402
from search.embedder import Embedder  # noqa: E402
from search.fusion import rrf  # noqa: E402
from search.indexes import BM25Index, DenseIndex  # noqa: E402

EVAL = DATA / "eval"
KS = (1, 3, 5, 10, 20)


def load_questions(n: int, seed: int = 0) -> list[dict]:
    qs = [q for q in json.loads((EVAL / "questions.json").read_text(encoding="utf-8")) if q["answerable"]]
    random.Random(seed).shuffle(qs)
    return qs[:n]


def build_chunks(count_tokens, size: int, overlap: int) -> tuple[list[str], list[str]]:
    texts, articles = [], []
    for pdf in sorted((EVAL / "pdfs").glob("*.pdf")):
        for c in chunk_pages(extract(pdf), count_tokens, size, overlap):
            texts.append(c.text)
            articles.append(pdf.stem)
    return texts, articles


def relevant_set(q: dict, texts: list[str], articles: list[str]) -> set[int]:
    golds = [a.lower() for a in q["answers"]]
    return {i for i, (t, a) in enumerate(zip(texts, articles))
            if a == q["article"] and any(g in t.lower() for g in golds)}


def score(rankings: list[list[int]], rel: list[set[int]]) -> dict:
    out = {f"recall@{k}": round(float(np.mean([bool(set(r[:k]) & s) for r, s in zip(rankings, rel)])), 4) for k in KS}
    rr = []
    for r, s in zip(rankings, rel):
        first = next((i for i, c in enumerate(r[:10], 1) if c in s), None)
        rr.append(1 / first if first else 0.0)
    out["mrr@10"] = round(float(np.mean(rr)), 4)
    return out


def run(questions, texts, articles, embedder: Embedder | None, modes, reranker=None, depth: int = 30) -> dict:
    rel = [relevant_set(q, texts, articles) for q in questions]
    keep = [i for i, s in enumerate(rel) if s]                 # drop questions whose answer got split by chunking
    questions, rel = [questions[i] for i in keep], [rel[i] for i in keep]
    ids = list(range(len(texts)))
    bm25 = BM25Index()
    bm25.set_all(ids, texts)
    dense = None
    if embedder:
        dense = DenseIndex(embedder.dim)
        dense.add(ids, embedder.embed_passages(texts))
    results = {}
    for mode in modes:
        t0 = time.perf_counter()
        rankings = []
        for q in questions:
            b = [i for i, _ in bm25.search(q["question"], depth)] if mode != "dense" else []
            d = [i for i, _ in dense.search(embedder.embed_query(q["question"]), depth)] if mode != "bm25" else []
            r = {"bm25": b, "dense": d}.get(mode) or [i for i, _ in rrf([d, b])][:depth]
            if mode == "hybrid+rerank":
                sc = reranker.scores(q["question"], [texts[i] for i in r])
                r = [i for _, i in sorted(zip(sc, r), key=lambda x: -x[0])]
            rankings.append(r)
        ms = (time.perf_counter() - t0) / len(questions) * 1e3
        results[mode] = {**score(rankings, rel), "ms_per_query": round(ms, 1), "questions": len(questions)}
        print(f"  {mode:15s} {results[mode]}", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=2000)
    a = ap.parse_args()
    questions = load_questions(a.questions)
    bge = Embedder("BAAI/bge-small-en-v1.5")

    print("1) chunk size sweep (hybrid BM25 + BGE)")
    sweep = {}
    for size in (128, 256, 384):
        texts, arts = build_chunks(bge.count_tokens, size, overlap=size // 6)
        print(f" size {size}: {len(texts)} chunks")
        res = run(questions, texts, arts, bge, ["hybrid"])["hybrid"]
        sweep[str(size)] = {**res, "chunks": len(texts)}
    update_metrics("retrieval", "chunk_size_sweep", sweep)

    print("2) retriever comparison at 256 tokens")
    texts, arts = build_chunks(bge.count_tokens, 256, 40)
    comp = run(questions, texts, arts, bge, ["bm25", "dense", "hybrid", "hybrid+rerank"],
               reranker=Reranker())
    comp = {("dense-bge" if k == "dense" else k): v for k, v in comp.items()}
    mini = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    comp["dense-minilm"] = run(questions, texts, arts, mini, ["dense"])["dense"]
    update_metrics("retrieval", "comparison_256", {"chunks": len(texts), **comp})


if __name__ == "__main__":
    main()
