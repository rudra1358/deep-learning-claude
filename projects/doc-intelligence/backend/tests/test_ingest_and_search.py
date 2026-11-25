"""Unit tests for the pure parts: extraction, chunking, fusion and the two indexes."""
import numpy as np
import pymupdf
import pytest

from ingest.chunk import chunk_pages, split_sentences
from ingest.extract import Page, clean, extract
from search.fusion import rrf
from search.indexes import BM25Index, DenseIndex, tokenize


def test_clean_fixes_pdf_artefacts():
    assert clean("an exam-\nple of\nwrapped   text", {"example"}) == "an example of wrapped text"


def test_line_hyphen_kept_for_real_compounds():
    """Regression: always gluing turned 'Anglo-Saxons' into 'AngloSaxons' (3.6 % of answers lost)."""
    vocab = {"example", "anglo", "saxons"}
    assert clean("the Anglo-\nSaxons", vocab) == "the Anglo-Saxons"
    assert clean("worst-\ncase", vocab) == "worst-case"
    assert clean("an exam-\nple", vocab) == "an example"
    # some PDF generators encode every hyphen as U+00AD (soft hyphen)
    assert clean("the Anglo­Saxons", vocab) == "the Anglo-Saxons"
    assert clean("an exam­\nple", vocab) == "an example"


def test_clean_normalises_unicode_spaces_and_ligatures():
    """Regression: some PDF fonts extract every space as U+00A0, which broke all phrase matching."""
    assert clean("William\xa0the\xa0Conqueror") == "William the Conqueror"
    assert clean("ﬁnal ofﬁce") == "final office"


def test_split_sentences_keeps_offsets():
    text = "First one. Second one! Is it third? Yes."
    sents = split_sentences(text)
    assert [s for _, s in sents] == ["First one.", "Second one!", "Is it third?", "Yes."]
    assert all(text[o:o + len(s)] == s for o, s in sents)


def count_words(t):
    return len(t.split())


def test_chunks_respect_budget_overlap_and_pages():
    s = " ".join(f"Sentence number {i} is here." for i in range(10))       # 5 words each
    pages = [Page(1, s), Page(2, s)]
    chunks = chunk_pages(pages, count_words, max_tokens=20, overlap_tokens=5)
    assert all(count_words(c.text) <= 20 for c in chunks)
    # consecutive chunks share exactly one sentence (the 5-word overlap)
    first_sents = [x for _, x in split_sentences(chunks[1].text)]
    assert first_sents[0] in chunks[0].text
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 2
    assert any(c.page_start == 1 and c.page_end == 2 for c in chunks)       # a chunk may span a page break
    # every sentence of the input appears in some chunk
    joined = " ".join(c.text for c in chunks)
    assert all(f"Sentence number {i} is here." in joined for i in range(10))
    c = chunks[0]
    assert pages[c.page_start - 1].text[c.char_start:].startswith(c.text.split(" ")[0])


def test_single_long_sentence_is_kept_not_dropped():
    chunks = chunk_pages([Page(1, "word " * 100 + ".")], count_words, max_tokens=10, overlap_tokens=2)
    assert len(chunks) == 1


def test_extract_pdf_docx_txt(tmp_path):
    pdf = tmp_path / "a.pdf"
    doc = pymupdf.open()
    for i in range(2):
        doc.new_page().insert_text((72, 72), f"Hello from page {i + 1}.")
    doc.save(pdf)
    assert [(p.number, p.text) for p in extract(pdf)] == [(1, "Hello from page 1."), (2, "Hello from page 2.")]

    import docx
    d = docx.Document()
    d.add_paragraph("A Word paragraph.")
    d.save(tmp_path / "b.docx")
    assert extract(tmp_path / "b.docx")[0].text == "A Word paragraph."

    (tmp_path / "c.txt").write_text("Para one\nwrapped.\n\nPara two.", encoding="utf-8")
    assert extract(tmp_path / "c.txt")[0].text == "Para one wrapped.\nPara two."


def test_extract_rejects_textless_pdf_and_bad_types(tmp_path):
    pdf = tmp_path / "scan.pdf"
    d = pymupdf.open()
    d.new_page()
    d.save(pdf)
    with pytest.raises(ValueError, match="no text found"):
        extract(pdf)
    (tmp_path / "x.exe").write_bytes(b"MZ")
    with pytest.raises(ValueError, match="unsupported"):
        extract(tmp_path / "x.exe")


def test_rrf_rewards_agreement():
    fused = dict(rrf([[1, 2, 3], [3, 1, 4]]))
    assert max(fused, key=fused.get) == 1             # ranks 1 and 2 beats ranks 3 and 1
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert 4 in fused                                 # found by one retriever only: still included


def test_dense_index_add_search_remove():
    idx = DenseIndex(3)
    vecs = np.eye(3, dtype=np.float32)
    idx.add([10, 20, 30], vecs)
    assert idx.search(np.array([0, 1, 0], np.float32), 1) == [(20, pytest.approx(1.0))]
    idx.remove([20])
    assert 20 not in [i for i, _ in idx.search(np.array([0, 1, 0], np.float32), 3)]


def test_bm25_prefers_rare_exact_terms():
    b = BM25Index()
    b.set_all([1, 2, 3], ["the contract ends in March", "payment is due monthly",
                          "clause 14b covers termination of the contract"])
    assert b.search("what does clause 14b say", 3)[0][0] == 3
    assert tokenize("What is THE clause?") == ["clause"]
    assert b.search("the of and", 3) == []            # only stop-words -> nothing to match


def test_bm25_common_word_still_scores_positive():
    """Regression test: textbook BM25 IDF is negative for words in > half the chunks, so a query
    for a very common word in a small collection returned nothing. The Lucene IDF never is."""
    b = BM25Index()
    b.set_all([1, 2, 3], ["the moon landing", "moon rocks", "bread recipe"])
    hits = b.search("moon", 3)
    assert {i for i, _ in hits} == {1, 2} and all(s > 0 for _, s in hits)


def test_bm25_score_matches_formula():
    import math
    b = BM25Index()
    b.set_all([1, 2], ["apple apple pie", "banana split"])
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    avg = 2.5
    tf_part = 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / avg))
    assert b.search("apple", 1)[0][1] == pytest.approx(idf * tf_part)
