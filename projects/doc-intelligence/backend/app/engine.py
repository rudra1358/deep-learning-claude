"""The engine: holds the models and indexes and implements search and question answering.

  search(question)  = retrieve (BM25 + dense, fused with RRF) -> optional cross-encoder re-rank
  ask(question)     = search -> extractive QA over the top chunks  (or)  LLM answer with citations

One Engine instance lives for the whole process. Models are loaded lazily the first time they
are needed (and only once, guarded by a lock), so the API starts quickly and a machine without
a GPU never loads the 1.5 B-parameter generator unless someone asks for a generative answer.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

from sqlalchemy import select

from ingest.chunk import chunk_pages
from search.embedder import Embedder
from search.fusion import rrf
from search.indexes import BM25Index, DenseIndex

from .config import get_settings
from .db import ChunkRow, Document, session_factory


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    score: float
    text: str = ""
    filename: str = ""
    page_start: int = 0
    page_end: int = 0
    char_start: int = 0
    ranks: dict = field(default_factory=dict)        # rank in each retriever (for the "why" view in the UI)


class Engine:
    def __init__(self) -> None:
        self.s = get_settings()
        self.embedder = Embedder(self.s.embedder)
        self.dense = DenseIndex(self.embedder.dim)
        self.bm25 = BM25Index()
        self.chunk_text: dict[int, str] = {}         # chunk id -> text      (kept in memory: fast BM25 rebuilds)
        self.chunk_doc: dict[int, int] = {}          # chunk id -> document id (for filtering by document)
        self._models: dict[str, object] = {}
        self._model_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        self.index_path = self.s.data_dir / "dense.faiss"

    # ------------------------------------------------------------------ lazy models
    def model(self, name: str):
        """Load a model once, on first use ("double-checked locking").

        One lock PER MODEL, not one global lock: with a single lock, the background worker loading
        the summariser (slow - it was still downloading) blocked every /ask request that needed the
        re-ranker for minutes. Now different models load independently; two requests for the SAME
        model still wait for one load instead of loading it twice."""
        if name not in self._models:
            with self._lock:                                  # short: only to get/create this model's lock
                lock = self._model_locks.setdefault(name, threading.Lock())
            with lock:
                if name not in self._models:
                    self._models[name] = self._load(name)
        return self._models[name]

    def _load(self, name: str):
        s, models_dir = self.s, self.s.models_dir
        if name == "reranker":
            from nlp.rerank import Reranker
            return Reranker(s.reranker, s.device)
        if name == "qa":
            from nlp.qa import ExtractiveQA
            return ExtractiveQA(s.qa_model, s.device)
        if name == "generator":
            if not s.enable_generator:
                raise RuntimeError("generative answers are disabled (DI_ENABLE_GENERATOR=false)")
            from nlp.generate import Generator
            return Generator(s.generator, s.device)
        if name == "summarizer":
            from nlp.summarize import Summarizer
            return Summarizer(s.summarizer, s.device)
        if name == "ner":
            from nlp.ner import NER
            return NER(str(models_dir / "ner"), s.device)
        if name == "classifier":
            from nlp.classify import Classifier
            return Classifier(str(models_dir / "classifier"), s.device)
        raise KeyError(name)

    def loaded_models(self) -> list[str]:
        return ["embedder"] + sorted(self._models)

    @property
    def no_answer_margin(self) -> float:
        if self.s.no_answer_margin is not None:
            return self.s.no_answer_margin
        if self.s.metrics_file.exists():                  # tuned on SQuAD 2.0 dev (training/eval_qa.py)
            qa = json.loads(self.s.metrics_file.read_text(encoding="utf-8")).get("qa", {})
            return float(qa.get("tuned", {}).get("no_answer_margin", 0.0))
        return 0.0

    # ------------------------------------------------------------------ indexes
    def load(self) -> None:
        """At startup: rebuild BM25 from the database and load (or rebuild) the FAISS index."""
        with session_factory()() as db:
            rows = db.execute(select(ChunkRow.id, ChunkRow.document_id, ChunkRow.text)
                              .join(Document).where(Document.status == "ready")).all()
        self.chunk_text = {r.id: r.text for r in rows}
        self.chunk_doc = {r.id: r.document_id for r in rows}
        self.bm25.set_all(list(self.chunk_text), list(self.chunk_text.values()))
        self.dense = DenseIndex.load(self.index_path, self.embedder.dim)
        if self.dense.index.ntotal != len(rows):          # index file missing or out of sync -> rebuild
            self.dense = DenseIndex(self.embedder.dim)
            if rows:
                ids = list(self.chunk_text)
                self.dense.add(ids, self.embedder.embed_passages([self.chunk_text[i] for i in ids]))
            self.dense.save(self.index_path)

    def make_chunks(self, pages):
        return chunk_pages(pages, self.embedder.count_tokens, self.s.chunk_tokens, self.s.chunk_overlap)

    def add_chunks(self, rows: list[ChunkRow]) -> None:
        vecs = self.embedder.embed_passages([r.text for r in rows])
        self.dense.add([r.id for r in rows], vecs)
        for r in rows:
            self.chunk_text[r.id], self.chunk_doc[r.id] = r.text, r.document_id
        self.bm25.set_all(list(self.chunk_text), list(self.chunk_text.values()))
        self.dense.save(self.index_path)

    def remove_chunks(self, ids: list[int]) -> None:
        self.dense.remove(ids)
        for i in ids:
            self.chunk_text.pop(i, None)
            self.chunk_doc.pop(i, None)
        self.bm25.set_all(list(self.chunk_text), list(self.chunk_text.values()))
        self.dense.save(self.index_path)

    # ------------------------------------------------------------------ search
    def retrieve(self, query: str, k: int, mode: str = "hybrid", document_ids: list[int] | None = None) -> list[Hit]:
        allowed = set(document_ids) if document_ids else None
        # ask each retriever for more than k, because the document filter may drop some results
        fetch = k * 4 if allowed else k * 2
        dense = bm25 = []
        if mode in ("dense", "hybrid"):
            dense = self.dense.search(self.embedder.embed_query(query), fetch)
        if mode in ("bm25", "hybrid"):
            bm25 = self.bm25.search(query, fetch)
        if allowed:
            dense = [h for h in dense if self.chunk_doc.get(h[0]) in allowed]
            bm25 = [h for h in bm25 if self.chunk_doc.get(h[0]) in allowed]
        if mode == "dense":
            ranked = dense
        elif mode == "bm25":
            ranked = bm25
        else:
            ranked = rrf([[i for i, _ in dense], [i for i, _ in bm25]])
        d_rank = {cid: r for r, (cid, _) in enumerate(dense, 1)}
        b_rank = {cid: r for r, (cid, _) in enumerate(bm25, 1)}

        def ranks(cid: int) -> dict:          # only retrievers that actually ran (None = ran, didn't find it)
            out = {}
            if mode in ("dense", "hybrid"):
                out["dense"] = d_rank.get(cid)
            if mode in ("bm25", "hybrid"):
                out["bm25"] = b_rank.get(cid)
            return out

        return [Hit(cid, self.chunk_doc[cid], float(sc), ranks=ranks(cid)) for cid, sc in ranked[:k] if cid in self.chunk_doc]

    def search(self, query: str, mode: str = "hybrid", rerank: bool = True, top_k: int | None = None,
               document_ids: list[int] | None = None) -> tuple[list[Hit], dict]:
        top_k = top_k or self.s.rerank_k
        t0 = time.perf_counter()
        hits = self.retrieve(query, self.s.retrieve_k if rerank else top_k, mode, document_ids)
        t1 = time.perf_counter()
        if rerank and hits:
            scores = self.model("reranker").scores(query, [self.chunk_text[h.chunk_id] for h in hits])
            for rank, (h, sc) in enumerate(zip(hits, scores), start=1):
                h.ranks["retrieval"] = rank            # where retrieval had it, before re-ranking
                h.score = sc
            hits = sorted(hits, key=lambda h: -h.score)
        hits = hits[:top_k]
        t2 = time.perf_counter()
        self._fill(hits)
        return hits, {"retrieve_ms": round((t1 - t0) * 1e3, 1), "rerank_ms": round((t2 - t1) * 1e3, 1)}

    def _fill(self, hits: list[Hit]) -> None:
        if not hits:
            return
        with session_factory()() as db:
            rows = {r.id: r for r in db.scalars(select(ChunkRow).where(ChunkRow.id.in_([h.chunk_id for h in hits])))}
            names = dict(db.execute(select(Document.id, Document.filename)).all())
        for h in hits:
            r = rows[h.chunk_id]
            h.text, h.page_start, h.page_end, h.char_start = r.text, r.page_start, r.page_end, r.char_start
            h.filename = names.get(h.document_id, "")

    # ------------------------------------------------------------------ answering
    def ask(self, question: str, mode: str = "extractive", document_ids: list[int] | None = None,
            history: list[dict] | None = None) -> dict:
        hits, timing = self.search(question, document_ids=document_ids)
        t0 = time.perf_counter()
        if not hits:
            return {"answer": None, "found": False, "sources": [], "timing": timing, "mode": mode}
        if mode == "generative":
            from nlp.generate import NOT_FOUND
            passages = [h.text for h in hits]
            # Answerability gate: the small LLM invents answers to most unanswerable questions (measured),
            # while the extractive reader is good at saying "no answer". If the reader finds nothing
            # (score <= 0), don't call the LLM at all.
            if self.s.generator_gate and self.model("qa").answer(question, passages, no_answer_margin=0.0) is None:
                result = {"answer": NOT_FOUND, "found": False, "cited": [], "highlight": None,
                          "gated": True}
            else:
                text, cited = self.model("generator").answer(question, passages, history)
                found = NOT_FOUND.lower()[:30] not in text.lower()
                result = {"answer": text, "found": found, "cited": cited, "highlight": None}
        else:
            # Three outcomes, by the reader's confidence (best span score - "no answer" score):
            #   > tuned margin  -> answer
            #   0 .. margin     -> "not found with confidence", but show the best candidate as a clearly
            #                      labelled low-confidence suggestion the user can check in its source
            #   <= 0            -> the model itself prefers "no answer": nothing
            # The margin trades wrong answers against missed ones; the suggestion keeps a missed-but-correct
            # answer one click away without claiming it. (Metrics count only real answers.)
            span = self.model("qa").answer(question, [h.text for h in hits], no_answer_margin=0.0)
            margin = self.no_answer_margin
            result = {"answer": None, "found": False, "cited": [], "highlight": None, "suggestion": None}
            if span is not None:
                highlight = {"source": span.passage + 1, "start": span.char_start, "end": span.char_end}
                if span.score > margin:
                    result.update(answer=span.text, found=True, cited=[span.passage + 1],
                                  confidence=round(span.score, 2), highlight=highlight)
                else:
                    result["suggestion"] = {"text": span.text, "confidence": round(span.score, 2),
                                            "threshold": round(margin, 2), "highlight": highlight}
        timing["answer_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        result.update(mode=mode, timing=timing, sources=[hit_json(h, i) for i, h in enumerate(hits, 1)])
        return result


def hit_json(h: Hit, n: int) -> dict:
    return {"n": n, "chunk_id": h.chunk_id, "document_id": h.document_id, "filename": h.filename,
            "page_start": h.page_start, "page_end": h.page_end, "char_start": h.char_start,
            "score": round(h.score, 4), "ranks": h.ranks, "text": h.text}


_engine: Engine | None = None
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = Engine()
                _engine.load()
    return _engine
