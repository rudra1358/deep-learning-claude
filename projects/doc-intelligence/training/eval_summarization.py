"""Measure summary quality with ROUGE on CNN/DailyMail (news articles + human-written highlights).

ROUGE-1 / ROUGE-2 = overlap of single words / word pairs with the reference summary;
ROUGE-L = longest common subsequence (rewards keeping the right order). F-measures, with
stemming ("running" ~ "run"). ROUGE rewards word overlap, not truth - so it is a sanity check
and a way to compare systems, not a guarantee the summary is faithful.

Baseline: LEAD-3 = just the first three sentences. On news it is famously hard to beat (news is
written with the key facts first), so a summariser is only useful if it gets close or above.

    python training/eval_summarization.py [--n 200]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time

import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer

from common import REPORTS, ROOT, update_metrics

sys.path.insert(0, str(ROOT / "backend"))
from nlp.summarize import Summarizer  # noqa: E402


def lead3(text: str) -> str:
    return " ".join(re.split(r"(?<=[.!?])\s+", text)[:3])


def sentences_per_line(text: str) -> str:
    """ROUGE-Lsum computes LCS sentence by sentence and finds sentences by NEWLINES. Without this step
    it silently degrades to plain ROUGE-L (our first run reported 30.5 instead of the real value)."""
    return "\n".join(s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip())


def rouge(preds: list[str], refs: list[str]) -> dict:
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    scores = [sc.score(sentences_per_line(r), sentences_per_line(p)) for p, r in zip(preds, refs)]
    return {k: round(100 * float(np.mean([s[k].fmeasure for s in scores])), 2) for k in ("rouge1", "rouge2", "rougeLsum")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--regenerate", action="store_true", help="ignore saved summaries and run the model again")
    a = ap.parse_args()
    rows = list(load_dataset("abisee/cnn_dailymail", "3.0.0", split="test"))
    random.Random(0).shuffle(rows)
    rows = rows[:a.n]
    arts, refs = [r["article"] for r in rows], [r["highlights"].replace("\n", " ") for r in rows]
    cache = REPORTS / "summaries_cnn_dm.json"            # generated summaries, so re-scoring needs no GPU
    if cache.exists() and "--regenerate" not in sys.argv:
        saved = json.loads(cache.read_text(encoding="utf-8"))
        preds, ms = saved["predictions"], saved["ms_per_article"]
    else:
        summ = Summarizer()
        t0 = time.perf_counter()
        preds = []
        for i in range(0, len(arts), 8):
            preds += summ.summarize_texts(arts[i:i + 8])
        ms = (time.perf_counter() - t0) / len(arts) * 1e3
        cache.write_text(json.dumps({"predictions": preds, "ms_per_article": ms}), encoding="utf-8")
    res = {"articles": len(arts), "distilbart": {**rouge(preds, refs), "ms_per_article": round(ms, 1)},
           "lead3_baseline": rouge([lead3(t) for t in arts], refs),
           "example": {"reference": refs[0], "distilbart": preds[0]}}
    print(res)
    update_metrics("summarization", "cnn_dailymail", res)


if __name__ == "__main__":
    main()
