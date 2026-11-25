"""Step 1 of ingestion: get plain text out of a file, page by page.

Why keep page numbers? Every answer must cite its source ("page 4 of contract.pdf"), so the
page a piece of text came from has to travel with it through the whole pipeline.

  PDF  -> PyMuPDF reads the text layer of each page (scanned PDFs without a text layer would
          need OCR - out of scope, we report "no text found" instead of failing silently)
  DOCX -> python-docx paragraphs. Word files have no fixed pages, so we start a new "page"
          every ~3000 characters to keep citations useful.
  TXT  -> same splitting as DOCX.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}
PSEUDO_PAGE_CHARS = 3000


@dataclass
class Page:
    number: int      # 1-based, as a human would say it
    text: str


_WORD = re.compile(r"\w+")
_LINE_HYPHEN = re.compile(r"(\w+)-\n(\w+)")


def vocabulary(texts: list[str]) -> set[str]:
    """All lower-cased words of a document - used to decide how to repair line-break hyphens."""
    return {w.lower() for t in texts for w in _WORD.findall(unicodedata.normalize("NFKC", t))}


def clean(text: str, vocab: set[str] | None = None) -> str:
    """Undo common PDF artefacts: odd Unicode, words hyphenated across lines, hard line breaks.

    1. NFKC normalisation maps "compatibility" characters to their plain form: non-breaking spaces
       (U+00A0) -> normal spaces, ligatures ("ﬁ" -> "fi"). Without it, a PDF whose font encodes
       spaces as non-breaking spaces gave text where "William the Conqueror" could not be found -
       search and answer matching silently failed for half of the evaluation questions.
    2. A hyphen at the end of a line is ambiguous: "exam-/ple" should become "example", but
       "Anglo-/Saxons" and "worst-/case" must keep their hyphen. We let the document decide: glue
       the parts only if the joined word appears elsewhere in the same document (`vocab`);
       otherwise keep the hyphen.
    3. Some PDF generators store EVERY hyphen as U+00AD ("soft hyphen"). Deleting it turned
       "Anglo-Saxons" into "AngloSaxons" (3.6 % of evaluation answers lost). So: a soft hyphen at a
       line end is treated like a line-break hyphen (rule 2); anywhere else it is a visible "-".
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­\n", "-\n").replace("­", "-")      # ­ = soft hyphen
    vocab = vocab or set()

    def fix(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        return a + b if (a + b).lower() in vocab else f"{a}-{b}"

    text = _LINE_HYPHEN.sub(fix, text)
    text = re.sub(r"\s*\n\s*", " ", text)                   # line breaks inside paragraphs -> space
    return re.sub(r"[ \t]+", " ", text).strip()


def _split_into_pages(paragraphs: list[str]) -> list[Page]:
    pages, buf = [], ""
    for p in paragraphs:
        if buf and len(buf) + len(p) > PSEUDO_PAGE_CHARS:
            pages.append(Page(len(pages) + 1, buf.strip()))
            buf = ""
        buf += p + "\n"
    if buf.strip():
        pages.append(Page(len(pages) + 1, buf.strip()))
    return pages


def extract(path: str | Path) -> list[Page]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        import pymupdf
        with pymupdf.open(path) as doc:
            raw = [page.get_text("text") for page in doc]
        vocab = vocabulary(raw)
        pages = [Page(i + 1, clean(t, vocab)) for i, t in enumerate(raw)]
    elif ext == ".docx":
        import docx
        paragraphs = [p.text.strip() for p in docx.Document(str(path)).paragraphs if p.text.strip()]
        pages = _split_into_pages(paragraphs)
    elif ext in (".txt", ".md"):
        raw = path.read_text(encoding="utf-8", errors="replace")
        vocab = vocabulary([raw])
        pages = _split_into_pages([clean(p, vocab) for p in re.split(r"\n\s*\n", raw) if p.strip()])
    else:
        raise ValueError(f"unsupported file type {ext}; supported: {sorted(SUPPORTED)}")
    pages = [p for p in pages if p.text]
    if not pages:
        raise ValueError("no text found (is this a scanned PDF without a text layer?)")
    return pages
