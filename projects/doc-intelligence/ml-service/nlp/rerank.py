"""Cross-encoder re-ranking: the "second opinion" on the retriever's shortlist.

Bi-encoder (retrieval): question and chunk are encoded SEPARATELY -> fast (chunk vectors are
precomputed) but the model never sees question and chunk together.
Cross-encoder (this file): question and chunk go through the transformer TOGETHER as one input
"[CLS] question [SEP] chunk [SEP]", so attention can compare every question word with every
chunk word. Much more accurate - but it must run once per (question, chunk) pair, so it is only
affordable on a short list (we re-rank the top ~20-30 candidates, not the whole collection).

Retrieve-then-rerank = speed of the bi-encoder + accuracy of the cross-encoder.
"""
from __future__ import annotations

from .common import pick_device


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "auto") -> None:
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=pick_device(device), max_length=512)

    def scores(self, question: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self.model.predict([(question, p) for p in passages], batch_size=32,
                                                      show_progress_bar=False)]
