"""Document classification with OUR fine-tuned model (training/finetune_classifier.py).

Same pretrained encoder as NER (DistilBERT), different head: instead of one label per token, a
single label for the whole text, predicted from the [CLS] token's final vector. Classes are the
AG News topics: World, Sports, Business, Sci/Tech.

A document is longer than the 512-token input, so we classify its first few chunks and average
the probabilities (beginnings of documents are usually the most representative).
"""
from __future__ import annotations

import torch

from .common import pick_device


class Classifier:
    def __init__(self, model_dir: str, device: str = "auto") -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    @torch.inference_mode()
    def probabilities(self, texts: list[str]) -> list[dict[str, float]]:
        enc = self.tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt").to(self.device)
        probs = self.model(**enc).logits.softmax(-1).cpu().tolist()
        return [dict(zip(self.labels, p)) for p in probs]

    def classify_document(self, chunks: list[str], first_n: int = 4) -> tuple[str, float, dict[str, float]]:
        per_chunk = self.probabilities(chunks[:first_n])
        avg = {lb: sum(p[lb] for p in per_chunk) / len(per_chunk) for lb in self.labels}
        label = max(avg, key=avg.get)
        return label, avg[label], avg
