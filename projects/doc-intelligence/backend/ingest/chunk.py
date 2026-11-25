"""Step 2 of ingestion: cut pages into chunks - the unit we embed, retrieve and cite.

Why chunk at all? An embedding model squeezes a whole text into ONE vector. A vector for a
50-page contract is a blur of everything; a vector for one paragraph is specific enough to
match a question. Also, models have input limits (e.g. 256-512 tokens).

Why sentence-aware? Cutting at a fixed character count splits sentences in half, and a
half-sentence often loses the very fact a question asks about. So we pack WHOLE sentences
into a chunk until a token budget is reached.

Why overlap? If the answer sits right at a boundary, the sentence before it is usually needed
for context. Repeating the last ~1-2 sentences at the start of the next chunk means every
sentence appears together with its neighbour in at least one chunk.

Sizes are measured in TOKENS of the embedding model (not characters/words), because that is
what the model's input limit counts. `count_tokens` is passed in so this file has no model code.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from .extract import Page

# Split after . ! ? when followed by whitespace and an uppercase letter/digit/quote. Simple and
# good enough for prose; abbreviations like "Dr. Smith" occasionally split early (harmless here).
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


@dataclass
class Chunk:
    index: int        # position within the document
    text: str
    page_start: int
    page_end: int
    char_start: int   # where the chunk starts inside page_start's text (for highlighting)


def split_sentences(text: str) -> list[tuple[int, str]]:
    """Returns (start offset, sentence) pairs."""
    out, pos = [], 0
    for piece in _SENTENCE_END.split(text):
        start = text.find(piece, pos)
        out.append((start, piece))
        pos = start + len(piece)
    return [(s, t) for s, t in out if t.strip()]


def chunk_pages(pages: list[Page], count_tokens: Callable[[str], int],
                max_tokens: int = 256, overlap_tokens: int = 40) -> list[Chunk]:
    # 1. flatten into sentences that remember their page and offset
    sents: list[tuple[int, int, str, int]] = []           # (page, offset, text, tokens)
    for page in pages:
        for off, s in split_sentences(page.text):
            sents.append((page.number, off, s, count_tokens(s)))

    chunks: list[Chunk] = []
    i = 0
    while i < len(sents):
        # 2. greedily add sentences while within budget (always at least one sentence, even if long -
        #    the embedder truncates it, which is better than dropping text)
        j, used = i, 0
        while j < len(sents) and (j == i or used + sents[j][3] <= max_tokens):
            used += sents[j][3]
            j += 1
        group = sents[i:j]
        chunks.append(Chunk(len(chunks), " ".join(s[2] for s in group),
                            group[0][0], group[-1][0], group[0][1]))
        if j >= len(sents):
            break
        # 3. step back over the last sentences worth ~overlap_tokens, so they start the next chunk
        back, k = 0, j
        while k - 1 > i and back + sents[k - 1][3] <= overlap_tokens:
            k -= 1
            back += sents[k][3]
        i = k
    return chunks
