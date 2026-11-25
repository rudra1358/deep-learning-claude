"""Named-entity recognition with OUR fine-tuned model (training/finetune_ner.py).

The model labels every token with a BIO tag: B-PER = first token of a person name, I-PER =
inside the same name, O = not an entity (types: PER, ORG, LOC, MISC - the CoNLL-2003 set).
Words are split into sub-word tokens ("Washington" -> "Wash", "##ington"), so we:
  1. predict a tag per token,
  2. take the tag of each word's FIRST sub-token (the convention used during training),
  3. merge consecutive B-X, I-X, I-X ... words into one entity span with character offsets.
For a whole document we run this per chunk and count how often each entity appears.
"""
from __future__ import annotations

from collections import Counter

import torch

from .common import pick_device


class NER:
    def __init__(self, model_dir: str, device: str = "auto") -> None:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).to(self.device).eval()
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def entities(self, text: str) -> list[dict]:
        enc = self.tok(text, truncation=True, max_length=512, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        word_ids = enc.word_ids(0)
        pred = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0].argmax(-1).tolist()

        ents, cur, prev_word = [], None, None
        for t, w in enumerate(word_ids):
            if w is None or w == prev_word:              # special token, or later sub-token of same word
                if w is not None and cur is not None:
                    cur["end"] = offsets[t][1]           # extend span to cover the whole word
                continue
            prev_word = w
            tag = self.id2label[pred[t]]
            kind = tag[2:] if tag != "O" else None
            if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur["type"] != kind)):
                if cur:
                    ents.append(cur)
                cur = {"type": kind, "start": offsets[t][0], "end": offsets[t][1]}
            elif tag.startswith("I-") and cur is not None:
                cur["end"] = offsets[t][1]
            else:                                        # "O"
                if cur:
                    ents.append(cur)
                cur = None
        if cur:
            ents.append(cur)
        for e in ents:
            e["text"] = text[e["start"]:e["end"]]
        return ents

    def document_entities(self, chunks: list[str], top: int = 40) -> list[dict]:
        counts: Counter = Counter()
        for c in chunks:
            for e in self.entities(c):
                if len(e["text"]) > 1:
                    counts[(e["text"], e["type"])] += 1
        return [{"text": t, "type": k, "count": n} for (t, k), n in counts.most_common(top)]
