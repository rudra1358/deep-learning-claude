"""Extractive question answering: find the answer as an exact SPAN of a retrieved chunk.

Model: RoBERTa fine-tuned on SQuAD 2.0. For "[question] [chunk]" it outputs two scores per
token: how likely this token STARTS the answer and how likely it ENDS it.
  span score = start_logit[i] + end_logit[j]   for i <= j, span not too long, inside the chunk
  "no answer" score = start_logit[CLS] + end_logit[CLS]  (SQuAD 2.0 teaches the model to point at
  the special first token when the text does not contain the answer)

Why extractive instead of a chatbot? The answer is copied from the document, so it cannot be
invented ("hallucinated"), and we can highlight exactly where it came from.

Long chunks: the model reads at most 384 tokens, so longer inputs are cut into overlapping
windows (stride) and the best span over all windows wins.

`no_answer_margin`: we answer only if (best span score - no-answer score) > margin. The margin
is tuned on SQuAD 2.0 dev data (training/eval_qa.py) to maximise F1, instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .common import pick_device


@dataclass
class Span:
    text: str
    score: float           # best span score minus the no-answer score (higher = more confident)
    passage: int           # index of the passage it came from
    char_start: int
    char_end: int


class ExtractiveQA:
    def __init__(self, model_name: str = "deepset/roberta-base-squad2", device: str = "auto",
                 max_len: int = 384, stride: int = 128, max_answer_tokens: int = 30) -> None:
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()                  # fp16: half the memory, faster on GPU
        self.max_len, self.stride, self.max_answer_tokens = max_len, stride, max_answer_tokens

    @torch.inference_mode()
    def best_spans(self, question: str, passages: list[str]) -> list[Span]:
        """One best span per passage (may be empty text if the model prefers 'no answer')."""
        if not passages:
            return []
        enc = self.tok([question] * len(passages), passages, truncation="only_second", max_length=self.max_len,
                       stride=self.stride, return_overflowing_tokens=True, return_offsets_mapping=True,
                       padding=True, return_tensors="pt")
        window_to_passage = enc.pop("overflow_to_sample_mapping").tolist()
        offsets = enc.pop("offset_mapping").tolist()
        out = self.model(**{k: v.to(self.device) for k, v in enc.items()})
        starts, ends = out.start_logits.float().cpu().numpy(), out.end_logits.float().cpu().numpy()

        best: dict[int, Span] = {}
        for w, p in enumerate(window_to_passage):
            seq_ids = enc.sequence_ids(w)
            ctx = np.array([s == 1 for s in seq_ids])          # True for tokens of the passage
            null = starts[w, 0] + ends[w, 0]
            s_log = np.where(ctx, starts[w], -1e9)
            e_log = np.where(ctx, ends[w], -1e9)
            # top candidates only (20 x 20 pairs), then keep the best valid one
            cand_s = np.argsort(-s_log)[:20]
            cand_e = np.argsort(-e_log)[:20]
            top = None
            for i in cand_s:
                for j in cand_e:
                    if j < i or j - i + 1 > self.max_answer_tokens or not ctx[i] or not ctx[j]:
                        continue
                    sc = s_log[i] + e_log[j]
                    if top is None or sc > top[0]:
                        top = (sc, i, j)
            if top is None:
                continue
            sc, i, j = top
            c0, c1 = offsets[w][i][0], offsets[w][j][1]
            span = Span(passages[p][c0:c1].strip(), float(sc - null), p, c0, c1)
            if p not in best or span.score > best[p].score:
                best[p] = span
        return [best[p] for p in sorted(best)]

    def answer(self, question: str, passages: list[str], no_answer_margin: float = 0.0) -> Span | None:
        spans = [s for s in self.best_spans(question, passages) if s.text]
        if not spans:
            return None
        top = max(spans, key=lambda s: s.score)
        return top if top.score > no_answer_margin else None
