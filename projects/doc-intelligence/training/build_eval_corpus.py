"""Build the labelled evaluation set from SQuAD 2.0 (dev split).

SQuAD 2.0 dev = 35 Wikipedia articles, each split into paragraphs, with ~12k crowd-written
questions. Every answerable question has the exact answer text; ~50 % are deliberately
UNANSWERABLE (they look relevant but the text does not contain the answer) - perfect for
testing whether the system says "not found" instead of inventing something.

We turn each article into a real multi-page PDF (title + paragraphs, A4, wrapped text), so the
evaluation runs through the exact same path as a user upload: PDF -> extract -> chunk -> embed.

Outputs:
  data/eval/pdfs/<article>.pdf
  data/eval/questions.json   [{id, article, question, answers: [...], answerable}]

Relevance label for retrieval: a retrieved chunk is "relevant" if it comes from the question's
article AND contains one of the gold answer strings (standard "answer recall" labelling).

    python training/build_eval_corpus.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf
from datasets import load_dataset

from common import DATA

OUT = DATA / "eval"
PAGE_W, PAGE_H, MARGIN = 595, 842, 56          # A4 in points
# The built-in PDF font (Helvetica) only covers Latin-1, so characters like "–", "−", "⊆" were lost
# and 1.3 % of gold answers vanished from the PDFs. A Unicode TrueType font keeps them.
FONT_FILES = [Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
FONT_FILE = next((str(f) for f in FONT_FILES if f.exists()), None)


def safe_name(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_")[:60]


def write_pdf(path, title: str, paragraphs: list[str]) -> int:
    doc = pymupdf.open()
    rect = pymupdf.Rect(MARGIN, MARGIN, PAGE_W - MARGIN, PAGE_H - MARGIN)
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = MARGIN
    blocks = [(title.replace("_", " "), 16)] + [(p, 10.5) for p in paragraphs]
    for text, size in blocks:
        while True:
            if rect.y1 - y < 3 * size:            # not even room for a few lines: next page
                page = doc.new_page(width=PAGE_W, height=PAGE_H)
                y = MARGIN
            box = pymupdf.Rect(rect.x0, y, rect.x1, rect.y1)
            # insert_textbox returns the unused height (>= 0) or a negative number if it did not fit
            left = page.insert_textbox(box, text, fontsize=size, fontname="uni" if FONT_FILE else "helv",
                                       fontfile=FONT_FILE)
            if left >= 0:
                y = rect.y1 - left + size * 0.8
                break
            if y == MARGIN:                       # paragraph longer than a full page: shrink font a bit
                size -= 0.5
                continue
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            y = MARGIN
    n = doc.page_count
    doc.save(path)
    doc.close()
    return n


def main() -> None:
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    (OUT / "pdfs").mkdir(parents=True, exist_ok=True)
    articles: dict[str, list[str]] = {}
    questions = []
    for row in ds:
        paras = articles.setdefault(row["title"], [])
        if row["context"] not in paras:
            paras.append(row["context"])
        questions.append({"id": row["id"], "article": safe_name(row["title"]), "question": row["question"].strip(),
                          "answers": sorted(set(row["answers"]["text"])), "answerable": bool(row["answers"]["text"])})
    pages = 0
    for title, paras in articles.items():
        pages += write_pdf(OUT / "pdfs" / f"{safe_name(title)}.pdf", title, paras)
    (OUT / "questions.json").write_text(json.dumps(questions), encoding="utf-8")
    n_ans = sum(q["answerable"] for q in questions)
    print(f"{len(articles)} articles -> {pages} PDF pages; {len(questions)} questions "
          f"({n_ans} answerable, {len(questions) - n_ans} unanswerable)")


if __name__ == "__main__":
    main()
