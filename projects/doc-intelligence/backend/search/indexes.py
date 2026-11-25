"""The two search indexes: dense (meaning) and BM25 (keywords).

DenseIndex - FAISS `IndexFlatIP`: stores every chunk vector and, for a query vector, computes
the inner product with ALL of them (exact search). With normalised vectors inner product =
cosine similarity. Exact search over 100k x 384 floats takes a few milliseconds; at millions
of chunks you would switch to an approximate index (IVF / HNSW) - same API, one line changes.
`IndexIDMap2` lets us use our database chunk ids as FAISS ids, so results map straight back
to rows and chunks of a deleted document can be removed.

BM25Index - classic keyword ranking (what search engines used before embeddings). A chunk
scores high when it contains the query's words, especially RARE words (IDF), with diminishing
returns for repeats and a penalty for very long chunks. Why keep it next to embeddings?
Embeddings are weak at exact tokens they never learned: product codes, names, numbers
("clause 14.2(b)"). BM25 is strong exactly there. Combining both (hybrid) beats either alone -
measured in training/eval_retrieval.py.
"""
from __future__ import annotations

import math
import re
import threading
from collections import Counter
from pathlib import Path

import faiss
import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of to in on for and or is are was were be been by with as at from that this it its "
            "what which who whom when where why how do does did".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class DenseIndex:
    def __init__(self, dim: int) -> None:
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.lock = threading.Lock()          # FAISS writes are not thread-safe

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        with self.lock:
            self.index.add_with_ids(vectors, np.asarray(ids, dtype=np.int64))

    def remove(self, ids: list[int]) -> None:
        with self.lock:
            self.index.remove_ids(np.asarray(ids, dtype=np.int64))

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        scores, ids = self.index.search(query[None, :], min(k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path, dim: int) -> DenseIndex:
        obj = cls(dim)
        if path.exists():
            obj.index = faiss.read_index(str(path))
        return obj


class BM25Index:
    """BM25 written out in full (it is short, and interviewers like to ask about it).

    For a query word w and a chunk d:
        idf(w)     = log(1 + (N - n_w + 0.5) / (n_w + 0.5))      N = #chunks, n_w = #chunks containing w
        tf part    = f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avg_len))   f = count of w in d
        score(d)   = sum over query words of idf(w) * tf part
    k1 = 1.2 controls how fast repeated words stop adding score (saturation);
    b = 0.75 controls how much long chunks are penalised (length normalisation).

    The "1 +" inside the log is the Lucene / Elasticsearch variant. The textbook version without
    it goes NEGATIVE for words in more than half the chunks - we hit exactly that bug with the
    `rank_bm25` library on a small collection (a query about "the Moon" in a Moon document found
    nothing), which is why this is implemented here instead.

    Rebuilt from scratch when documents change: ~1 s for 50k chunks, fine for uploads. At larger
    scale an inverted index in a search engine (Elasticsearch / OpenSearch) does this incrementally.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.ids: list[int] = []
        self.tf: list[Counter] = []              # word counts per chunk
        self.lengths = np.zeros(0)
        self.idf: dict[str, float] = {}
        self.lock = threading.Lock()

    def set_all(self, ids: list[int], texts: list[str]) -> None:
        tf = [Counter(tokenize(t)) for t in texts]
        df: Counter = Counter()                  # in how many chunks each word appears
        for c in tf:
            df.update(c.keys())
        n = len(tf)
        with self.lock:
            self.ids, self.tf = list(ids), tf
            self.lengths = np.array([sum(c.values()) for c in tf], dtype=np.float64)
            self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = [w for w in tokenize(query) if w in self.idf]
        if not q or not self.ids:
            return []
        norm = self.k1 * (1 - self.b + self.b * self.lengths / max(self.lengths.mean(), 1e-9))
        scores = np.zeros(len(self.ids))
        for w in set(q):
            f = np.array([c.get(w, 0) for c in self.tf], dtype=np.float64)
            scores += self.idf[w] * f * (self.k1 + 1) / (f + norm)
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]
