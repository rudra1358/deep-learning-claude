"""Fine-tune DistilBERT to classify documents by topic (AG News: World / Sports / Business /
Sci/Tech). Output: models/classifier/ (used by backend/nlp/classify.py).

We train on a seeded random 30,000 of the 120,000 training articles: on a laptop GPU that
takes minutes instead of an hour, and (measured) costs little accuracy - a good example of
checking whether more data is worth its cost. The full 7,600-article test set is used.

Reported: accuracy, macro-F1 (average of per-class F1, so every class counts equally) and the
confusion matrix (which topics get mixed up).

    python training/finetune_classifier.py
"""
from __future__ import annotations

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from common import MODELS, seed_everything, train_loop, update_metrics

BASE = "distilbert-base-cased"
EPOCHS, LR, BATCH, TRAIN_SIZE, MAX_LEN = 2, 3e-5, 32, 30_000, 128


def main() -> None:
    seed_everything(0)
    ds = load_dataset("fancyzhx/ag_news")
    labels = ds["train"].features["label"].names                     # ['World', 'Sports', 'Business', 'Sci/Tech']
    tok = AutoTokenizer.from_pretrained(BASE)
    enc = lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN)  # noqa: E731
    train = ds["train"].shuffle(seed=0).select(range(TRAIN_SIZE)).map(enc, batched=True, remove_columns=["text"])
    test = ds["test"].map(enc, batched=True, remove_columns=["text"])
    train, test = train.rename_column("label", "labels"), test.rename_column("label", "labels")
    collate = DataCollatorWithPadding(tok)
    train_loader = DataLoader(train, batch_size=BATCH, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test, batch_size=128, collate_fn=collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={name: i for i, name in enumerate(labels)}
    ).to(device)

    def predict(m):
        m.eval()
        ys, ps = [], []
        with torch.inference_mode():
            for b in test_loader:
                b = {k: v.to(device) for k, v in b.items()}
                ps += m(**b).logits.argmax(-1).tolist()
                ys += b["labels"].tolist()
        return np.array(ys), np.array(ps)

    def evaluate(m):
        y, p = predict(m)
        return {"test_accuracy": round(accuracy_score(y, p), 4), "test_macro_f1": round(f1_score(y, p, average="macro"), 4)}

    history = train_loop(model, train_loader, EPOCHS, LR, evaluate, "clf")
    y, p = predict(model)
    model.save_pretrained(MODELS / "classifier")
    tok.save_pretrained(MODELS / "classifier")
    update_metrics("classifier", "distilbert-agnews", {
        **evaluate(model), "labels": labels, "confusion_matrix": confusion_matrix(y, p).tolist(),
        "per_class_f1": dict(zip(labels, [round(v, 4) for v in f1_score(y, p, average=None)])),
        "history": history, "train_size": TRAIN_SIZE, "test_size": len(test)})
    print("saved models/classifier", evaluate(model))


if __name__ == "__main__":
    main()
