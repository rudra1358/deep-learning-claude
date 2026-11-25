"""Abstractive summarisation of whole documents (map-reduce).

Model: DistilBART fine-tuned on CNN/DailyMail (news article -> highlights). It reads at most
1024 tokens, but documents are longer. Map-reduce handles any length:

  map:    split the document into ~900-token sections and summarise each one
  reduce: join the section summaries and summarise THAT (repeat if it is still too long)

"Abstractive" = the model writes new sentences (vs extractive = copying sentences). Quality is
measured with ROUGE against human-written summaries in training/eval_summarization.py.
"""
from __future__ import annotations

import torch

from .common import pick_device


class Summarizer:
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6", device: str = "auto",
                 section_tokens: int = 900) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.section_tokens = section_tokens

    @torch.inference_mode()
    def summarize_texts(self, texts: list[str], max_len: int = 142, min_len: int = 30) -> list[str]:
        enc = self.tok(texts, truncation=True, max_length=1024, padding=True, return_tensors="pt").to(self.device)
        out = self.model.generate(**enc, num_beams=4, max_length=max_len, min_length=min_len,
                                  length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=True)
        return [s.strip() for s in self.tok.batch_decode(out, skip_special_tokens=True)]

    def _sections(self, text: str) -> list[str]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        step = self.section_tokens
        return [self.tok.decode(ids[i:i + step]) for i in range(0, len(ids), step)]

    def summarize(self, text: str, max_rounds: int = 3) -> str:
        for _ in range(max_rounds):
            sections = self._sections(text)
            if len(sections) == 1:
                return self.summarize_texts(sections)[0]
            summaries = []
            for i in range(0, len(sections), 4):                 # batches of 4 keep GPU memory small
                summaries += self.summarize_texts(sections[i:i + 4], max_len=100, min_len=20)
            text = " ".join(summaries)
        return self.summarize_texts([text])[0]
