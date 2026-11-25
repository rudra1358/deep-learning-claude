"""Fine-tune DistilBERT for named-entity recognition on CoNLL-2003 (news sentences labelled with
PER / ORG / LOC / MISC in BIO format). Output: models/ner/ (used by backend/nlp/ner.py).

The key preprocessing step is LABEL ALIGNMENT. CoNLL labels whole words, but the tokenizer
splits words into sub-words: "Washington" -> ["Wash", "##ington"]. We give the label to the
FIRST sub-word and -100 to the others; -100 tells PyTorch's cross-entropy to ignore that
position, so each word is counted exactly once in the loss and in the metric.

Metric: entity-level F1 (seqeval) - an entity counts as correct only if BOTH its boundaries
and its type are right. That is stricter (and more honest) than per-token accuracy, where the
many "O" tokens would make any model look good.

    python training/finetune_ner.py
"""
from __future__ import annotations

import ast
import re
import sys

import torch
from datasets import load_dataset
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from common import MODELS, REPORTS, seed_everything, train_loop, update_metrics

BASE = "distilbert-base-cased"      # cased: capital letters are a strong entity signal
EPOCHS, LR, BATCH = 3, 5e-5, 16


def history_from_log() -> list[dict]:
    """Per-epoch records printed by train_loop, recovered from reports/log_ner.txt."""
    log = REPORTS / "log_ner.txt"
    if not log.exists():
        return []
    rows = re.findall(r"^\[ner\] (\{.*\})$", log.read_text(encoding="utf-8", errors="replace"), re.M)
    return [ast.literal_eval(re.sub(r"np\.float64\(([^)]*)\)", r"\1", r)) for r in rows]


def main() -> None:
    seed_everything(0)
    # the repo's main branch is a loading *script*, which datasets >= 4 refuses to run; the Hub keeps
    # an auto-converted Parquet copy of the same data on this branch
    ds = load_dataset("eriktks/conll2003", revision="refs/convert/parquet")
    labels = ds["train"].features["ner_tags"].feature.names          # ['O', 'B-PER', 'I-PER', ...]
    tok = AutoTokenizer.from_pretrained(BASE)

    def encode(batch):
        enc = tok(batch["tokens"], is_split_into_words=True, truncation=True, max_length=256)
        enc["labels"] = []
        for i, tags in enumerate(batch["ner_tags"]):
            prev, row = None, []
            for w in enc.word_ids(i):
                row.append(-100 if w is None or w == prev else tags[w])   # label only first sub-token
                prev = w
            enc["labels"].append(row)
        return enc

    cols = ds["train"].column_names
    train = ds["train"].map(encode, batched=True, remove_columns=cols)
    test = ds["test"].map(encode, batched=True, remove_columns=cols)
    collate = DataCollatorForTokenClassification(tok)               # pads input_ids AND labels (with -100)
    train_loader = DataLoader(train, batch_size=BATCH, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test, batch_size=64, collate_fn=collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(
        BASE, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={name: i for i, name in enumerate(labels)}
    ).to(device)

    def evaluate(m, report: bool = False):
        m.eval()
        y_true, y_pred = [], []
        with torch.inference_mode():
            for b in test_loader:
                b = {k: v.to(device) for k, v in b.items()}
                pred = m(**b).logits.argmax(-1)
                for p_row, l_row in zip(pred.tolist(), b["labels"].tolist()):
                    y_true.append([labels[t] for t in l_row if t != -100])
                    y_pred.append([labels[p] for p, t in zip(p_row, l_row) if t != -100])
        out = {"test_f1": round(float(f1_score(y_true, y_pred)), 4)}
        if report:
            out["report"] = classification_report(y_true, y_pred, output_dict=True)
        return out

    if "--eval-only" in sys.argv:                 # re-score an already trained model (no training)
        model = AutoModelForTokenClassification.from_pretrained(MODELS / "ner").to(device)
        history = history_from_log()
    else:
        history = train_loop(model, train_loader, EPOCHS, LR, evaluate, "ner")
        model.save_pretrained(MODELS / "ner")
        tok.save_pretrained(MODELS / "ner")
    final = evaluate(model, report=True)
    per_type = {k: {m: round(float(v[m]), 4) for m in ("precision", "recall", "f1-score", "support")}
                for k, v in final["report"].items()}
    update_metrics("ner", "distilbert-conll2003", {"test_f1": final["test_f1"], "per_type": per_type,
                                                   "history": history, "train_sentences": len(train),
                                                   "test_sentences": len(test)})
    print("saved models/ner, test F1 =", final["test_f1"])


if __name__ == "__main__":
    main()
