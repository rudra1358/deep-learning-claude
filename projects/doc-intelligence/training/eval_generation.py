"""Measure the generative (LLM) answer mode on SQuAD 2.0 questions over the full PDF corpus.

Free-form answers are not judged with EM (the LLM writes sentences, not spans). We measure the
things that matter for a document assistant:
  answer_contains_gold  - (answerable) the written answer contains a gold answer string
  cited                 - the answer cites at least one valid source [n]
  refused_correctly     - (unanswerable) the model says it could not find the answer
                          instead of inventing one ("hallucination rate" = 100 - this)
Same retrieval as the API (hybrid + re-rank, top 5). Seeded sample, greedy decoding.

    python training/eval_generation.py [--n 150]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

from common import DATA, ROOT, update_metrics

sys.path.insert(0, str(ROOT / "backend"))
from ingest.chunk import chunk_pages  # noqa: E402
from ingest.extract import extract  # noqa: E402
from nlp.generate import NOT_FOUND, Generator  # noqa: E402
from nlp.qa import ExtractiveQA  # noqa: E402
from nlp.rerank import Reranker  # noqa: E402
from search.embedder import Embedder  # noqa: E402
from search.fusion import rrf  # noqa: E402
from search.indexes import BM25Index, DenseIndex  # noqa: E402

VARIANTS = {   # name -> (prompt version, use the reader as an answerability gate)
    "prompt_v1": ("v1", False),
    "prompt_v2": ("v2", False),
    "prompt_v2+reader_gate": ("v2", True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--variants", default="prompt_v2,prompt_v2+reader_gate",
                    help=f"comma-separated, from {list(VARIANTS)}")
    a = ap.parse_args()
    qs = json.loads((DATA / "eval" / "questions.json").read_text(encoding="utf-8"))
    random.Random(1).shuffle(qs)
    ans = [q for q in qs if q["answerable"]][: a.n // 2]
    unans = [q for q in qs if not q["answerable"]][: a.n // 2]

    emb, rer, gen, qa = Embedder("BAAI/bge-small-en-v1.5"), Reranker(), Generator(), ExtractiveQA()
    texts = []
    for pdf in sorted((DATA / "eval" / "pdfs").glob("*.pdf")):
        texts += [c.text for c in chunk_pages(extract(pdf), emb.count_tokens, 256, 40)]
    ids = list(range(len(texts)))
    bm25, dense = BM25Index(), DenseIndex(emb.dim)
    bm25.set_all(ids, texts)
    dense.add(ids, emb.embed_passages(texts))

    def sources(q: str) -> list[str]:
        cand = [i for i, _ in rrf([[i for i, _ in dense.search(emb.embed_query(q), 30)],
                                   [i for i, _ in bm25.search(q, 30)]])][:30]
        sc = rer.scores(q, [texts[i] for i in cand])
        return [texts[i] for _, i in sorted(zip(sc, cand), key=lambda x: -x[0])[:5]]

    retrieved = {q["id"]: sources(q["question"]) for q in ans + unans}     # same context for every variant

    for name in a.variants.split(","):
        prompt, gate = VARIANTS[name]
        gen.prompt = prompt
        stats = {"answerable": {"n": 0, "contains_gold": 0, "cited": 0, "refused": 0},
                 "unanswerable": {"n": 0, "refused": 0, "cited": 0}}
        examples, t0 = [], time.perf_counter()
        for kind, group in (("answerable", ans), ("unanswerable", unans)):
            for q in group:
                ctx = retrieved[q["id"]]
                if gate and qa.answer(q["question"], ctx, no_answer_margin=0.0) is None:
                    text, cited = NOT_FOUND, []           # reader says "no answer": the LLM is not called
                else:
                    text, cited = gen.answer(q["question"], ctx)
                refused = NOT_FOUND.lower()[:30] in text.lower()
                s = stats[kind]
                s["n"] += 1
                s["refused"] += refused
                s["cited"] += bool(cited)
                if kind == "answerable":
                    s["contains_gold"] += any(g.lower() in text.lower() for g in q["answers"])
                if len(examples) < 4 and (kind == "unanswerable" or len(examples) < 2):
                    examples.append({"question": q["question"], "gold": q["answers"], "answer": text})
        ms = (time.perf_counter() - t0) / (len(ans) + len(unans)) * 1e3
        A, U = stats["answerable"], stats["unanswerable"]
        res = {"model": "Qwen/Qwen2.5-1.5B-Instruct", "prompt": prompt, "reader_gate": gate,
               "questions": A["n"] + U["n"], "ms_per_question": round(ms),
               "answerable": {"answer_contains_gold": round(100 * A["contains_gold"] / A["n"], 1),
                              "cited_at_least_one_source": round(100 * A["cited"] / A["n"], 1),
                              "wrongly_refused": round(100 * A["refused"] / A["n"], 1)},
               "unanswerable": {"refused_correctly": round(100 * U["refused"] / U["n"], 1),
                                "hallucination_rate": round(100 * (U["n"] - U["refused"]) / U["n"], 1)},
               "examples": examples}
        print(name, json.dumps({k: v for k, v in res.items() if k != "examples"}), flush=True)
        update_metrics("generation", name, res)


if __name__ == "__main__":
    main()
