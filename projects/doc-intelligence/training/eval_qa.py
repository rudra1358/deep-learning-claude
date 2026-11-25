"""Measure question answering end to end on SQuAD 2.0 (answerable AND unanswerable questions).

Metrics (official SQuAD definitions):
  EM - exact match after normalisation (lower-case, remove punctuation and a/an/the, squash spaces)
  F1 - word-overlap F1 between predicted and gold answer (best over the gold answers)
  For an UNANSWERABLE question, the only correct output is "no answer" (EM = F1 = 1), anything
  else scores 0 - so a system that always answers is punished.

Two settings, to see WHERE errors come from:
  oracle   - the reader gets the gold paragraph (measures the QA model alone)
  system   - the real pipeline: hybrid retrieval + re-rank over all 35 articles, top 5 chunks
  The gap between them = the cost of retrieval.

The "no-answer margin" (answer only if span score - null score > margin) is TUNED on one half
of the questions and REPORTED on the other half - choosing a threshold on the same data you
report on would overstate the result.

    python training/eval_qa.py [--questions 2000]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import string
import sys
import time

import numpy as np
from datasets import load_dataset

from common import DATA, ROOT, update_metrics

sys.path.insert(0, str(ROOT / "ml-service"))
from ingest.chunk import chunk_pages  # noqa: E402
from ingest.extract import extract  # noqa: E402
from nlp.qa import ExtractiveQA  # noqa: E402
from nlp.rerank import Reranker  # noqa: E402
from search.embedder import Embedder  # noqa: E402
from search.fusion import rrf  # noqa: E402
from search.indexes import BM25Index, DenseIndex  # noqa: E402


# --- official SQuAD normalisation and scores -------------------------------------------------
def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def score_one(pred: str | None, golds: list[str]) -> tuple[float, float]:
    if not golds:                                   # unanswerable
        return (1.0, 1.0) if pred is None else (0.0, 0.0)
    if pred is None:
        return 0.0, 0.0
    return max(float(normalize(pred) == normalize(g)) for g in golds), max(f1(pred, g) for g in golds)


def evaluate(preds: list[tuple[str | None, float]], golds: list[list[str]], margin: float) -> dict:
    ems, f1s, ans_f1, noans_acc = [], [], [], []
    for (text, sc), g in zip(preds, golds):
        p = text if (text and sc > margin) else None
        em, f = score_one(p, g)
        ems.append(em)
        f1s.append(f)
        (ans_f1 if g else noans_acc).append(f)
    return {"EM": round(100 * np.mean(ems), 2), "F1": round(100 * np.mean(f1s), 2),
            "F1_answerable": round(100 * np.mean(ans_f1), 2), "accuracy_unanswerable": round(100 * np.mean(noans_acc), 2)}


def tune_margin(preds, golds) -> float:
    best = max(np.arange(-6, 6.01, 0.25), key=lambda m: evaluate(preds, golds, m)["F1"])
    return float(best)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=2000)
    a = ap.parse_args()
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    rows = list(ds)
    random.Random(0).shuffle(rows)
    rows = rows[:a.questions]
    golds = [sorted(set(r["answers"]["text"])) for r in rows]
    half = len(rows) // 2
    qa = ExtractiveQA()

    # --- oracle: gold paragraph only
    t0 = time.perf_counter()
    oracle = []
    for r in rows:
        s = qa.best_spans(r["question"], [r["context"]])
        oracle.append((s[0].text, s[0].score) if s else (None, -1e9))
    oracle_ms = (time.perf_counter() - t0) / len(rows) * 1e3

    # --- system: retrieve from all 35 articles (same index the API builds)
    emb, rer = Embedder("BAAI/bge-small-en-v1.5"), Reranker()
    texts = []
    for pdf in sorted((DATA / "eval" / "pdfs").glob("*.pdf")):
        texts += [c.text for c in chunk_pages(extract(pdf), emb.count_tokens, 256, 40)]
    ids = list(range(len(texts)))
    bm25, dense = BM25Index(), DenseIndex(emb.dim)
    bm25.set_all(ids, texts)
    dense.add(ids, emb.embed_passages(texts))
    t0 = time.perf_counter()
    system = []
    for r in rows:
        q = r["question"]
        cand = [i for i, _ in rrf([[i for i, _ in dense.search(emb.embed_query(q), 30)],
                                   [i for i, _ in bm25.search(q, 30)]])][:30]
        sc = rer.scores(q, [texts[i] for i in cand])
        top = [texts[i] for _, i in sorted(zip(sc, cand), key=lambda x: -x[0])[:5]]
        spans = [s for s in qa.best_spans(q, top) if s.text]
        best = max(spans, key=lambda s: s.score) if spans else None
        system.append((best.text, best.score) if best else (None, -1e9))
    system_ms = (time.perf_counter() - t0) / len(rows) * 1e3

    result = {"questions": len(rows), "answerable": sum(bool(g) for g in golds),
              "tune_split": half, "report_split": len(rows) - half}
    for name, preds, ms in (("oracle", oracle, oracle_ms), ("system", system, system_ms)):
        m = tune_margin(preds[:half], golds[:half])                       # tune on first half ...
        result[name] = {"no_answer_margin": m, "ms_per_question": round(ms, 1),
                        "tuned_on_first_half_reported_on_second": evaluate(preds[half:], golds[half:], m),
                        "margin_0_reported_on_second": evaluate(preds[half:], golds[half:], 0.0),
                        "always_answer_reported_on_second": evaluate(preds[half:], golds[half:], -1e8)}
        print(name, json.dumps(result[name], indent=1), flush=True)
    result["tuned"] = {"no_answer_margin": result["system"]["no_answer_margin"]}   # used by the API
    update_metrics("qa", "squad2", result)
    update_metrics("qa", "tuned", result["tuned"])


if __name__ == "__main__":
    main()
