"""Test setup: the real code paths with tiny FAKE models.

Real models are 20 MB - 3 GB and slow to load; their quality is measured by the evaluation
scripts in training/. These tests check the PLUMBING: index a file -> extract -> chunk -> embed
-> FAISS + BM25 + chunk store -> search/fusion/re-rank -> answer -> JSON. So we swap in:
  FakeEmbedder  - hashes words into a 256-dim vector (texts sharing words are similar)
  FakeReranker  - scores = number of shared words
  FakeQA        - "answers" with the sentence of the passage sharing most words with the question
  FakeSummarizer- first sentence
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np
import pytest

WORD = re.compile(r"[a-z0-9]+")


def words(t: str) -> list[str]:
    return WORD.findall(t.lower())


class FakeEmbedder:
    def __init__(self, *_a, **_k):
        self.name, self.dim, self.max_tokens = "fake", 256, 512

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        for w in words(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def embed_query(self, text):
        return self._vec(text)


class FakeReranker:
    def scores(self, q, passages):
        qs = set(words(q))
        return [float(len(qs & set(words(p)))) for p in passages]


class FakeQA:
    def answer(self, question, passages, no_answer_margin=0.0):
        from nlp.qa import Span
        qs = set(words(question))
        best = None
        for pi, p in enumerate(passages):
            for m in re.finditer(r"[^.]+\.", p):
                sc = len(qs & set(words(m.group())))
                if best is None or sc > best[0]:
                    best = (sc, pi, m.start(), m.end())
        if not best or best[0] < 2:                 # fewer than 2 shared words -> "no answer"
            return None
        sc, pi, s, e = best
        return Span(passages[pi][s:e].strip(), float(sc), pi, s, e)


class FakeSummarizer:
    def summarize(self, text):
        return text.split(".")[0] + "."


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("di")
    os.environ.update({"DI_DATA_DIR": str(tmp / "data"), "DI_MODELS_DIR": str(tmp / "models"),
                       "DI_METRICS_FILE": str(tmp / "metrics.json"), "DI_ENABLE_GENERATOR": "false",
                       "DI_CHUNK_TOKENS": "40", "DI_CHUNK_OVERLAP": "8"})
    from app import engine as engine_mod

    engine_mod.Embedder = FakeEmbedder
    fakes = {"reranker": FakeReranker(), "qa": FakeQA(), "summarizer": FakeSummarizer()}
    real_load = engine_mod.Engine._load

    def fake_load(self, name):
        return fakes[name] if name in fakes else real_load(self, name)

    engine_mod.Engine._load = fake_load
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c
