## 7. What the numbers mean

**Retrieval.** BM25 beat dense embeddings at rank 1 (80.8 vs 65.0 Recall@1). This is a property of SQuAD:
crowdworkers wrote the questions while reading the paragraph, so questions reuse its exact words — ideal for
keyword search. On paraphrased real-world questions dense retrieval usually wins, so neither is chosen alone.
Hybrid (RRF) gives the best deep recall (99.2 % Recall@20 = the pool handed to the re-ranker) and the
cross-encoder turns that into the best rank-1 accuracy (85.1 %, MRR 0.903) for ~0.1 s extra per query.

**Reader vs retrieval.** The reader alone (gold paragraph given) scores F1 83.4 — in line with the published
~83 for `roberta-base-squad2`, which also validates our hand-written span decoding. The full system scores F1
77.6: retrieval costs 5.8 F1 points. Tuning the no-answer margin on held-out questions added +3.6 F1 for the
full system; it needs a stricter margin (+2.75) than the reader alone (+0.75) because 35 documents offer many
more plausible-looking wrong spans. Never saying "no answer" would halve F1 (40.2).

**Summaries.** DistilBART beats the strong LEAD-3 baseline on all ROUGE scores. ROUGE measures overlap, not
truth: the first evaluated summary says "Nico Rosberg finished 2nd with Mercedes team-mate Lewis Hamilton 2nd"
although Hamilton won — a fluent, high-overlap, factually wrong sentence.
(ROUGE-Lsum needs one sentence per line; the first run without that reported 30.5 instead of 40.7.)

**Error example — a confident wrong answer.** "Who patented the first steam engine?" → "Thomas Newcomen"
(reader score +5.6, above threshold). The document says Jerónimo de Ayanz y Beaumont got the first patent in
1606. Retrieval had the right chunk at rank 2; the reader preferred "invented by Thomas Newcomen" from a sentence
about the first *commercially successful* engine. The reader scores each passage independently, so scores are
not calibrated across passages.

**Topic labels.** On AG News the classifier reaches 93.0 % accuracy; its main confusion is Business ↔ Sci/Tech (140 + 113 errors) — tech-industry news overlaps both. It knows only these four topics; a history article (Normans) was labelled
"Sci/Tech" at 49 % — the label set must match the use case, and low-confidence labels should be flagged.

**Generative mode — prompt engineering vs architecture.** With instructions only (v1), the 1.5 B LLM invented
an answer for 86.7 % of unanswerable questions and cited a source in only 24 % of answers. A stricter prompt
with two worked examples (v2) improved answers (78.7 % contain the gold answer) and citations (40 %) — but made
hallucination *worse* (97.3 %): the examples made the model more eager to answer everything. The fix was
architectural, not more prompting: the extractive reader acts as an **answerability gate** — if it finds no
answer in the sources, the LLM is not called. Hallucination fell from 97.3 % to 24.0 %, at the cost of 8 % of
answerable questions refused and 6.7 points of contains-gold; it is also 38 % faster (the LLM is skipped for
about half the questions). Example: "In what meeting did Shirley lay out plans for **1765**?" (the document says
1756) — the LLM alone "corrected" the date and answered; with the gate the system says it could not find it.
Citations remain the weak spot of a 1.5 B model (~39 %); a larger model or constrained decoding would be next.
