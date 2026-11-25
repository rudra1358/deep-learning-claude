"""Reciprocal Rank Fusion (RRF): merge several ranked lists into one.

Problem: BM25 scores (e.g. 0-30) and cosine similarities (0-1) are on different scales, so you
cannot just add them. RRF ignores the raw scores and uses only each item's RANK in each list:

    score(chunk) = sum over lists of  1 / (k + rank_in_that_list)          (rank starts at 1)

A chunk ranked high by both retrievers wins; a chunk found by only one still gets credit.
k = 60 is the value from the original paper (Cormack et al., 2009); it dampens the gap between
rank 1 and rank 2 so one list cannot dominate.
"""
from __future__ import annotations


def rrf(ranked_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
