# Interview questions on this project — with model answers

Learn the ideas, then answer in your own words. Quote numbers only from [RESULTS.md](RESULTS.md).

## The project

**1. Walk me through your project.**
Problem (grounded answers over private documents, with sources and honest "not found") → ingestion (page-aware
extraction, cleaning, sentence-aware chunks with overlap) → hybrid retrieval (BM25 + BGE in FAISS, RRF) →
cross-encoder re-ranking → extractive reader with a tuned no-answer threshold, or a local LLM with citations →
fine-tuned NER + topic classifier, map-reduce summaries → a Python (FastAPI) ML service behind an Express +
MongoDB server with background ingestion jobs → React UI with highlighted sources → evaluated on SQuAD 2.0 as PDFs. Finish with one limitation and the next step.

**2. What is RAG and why not just fine-tune an LLM on the documents?**
Retrieval-Augmented Generation = retrieve relevant passages at question time and give them to the model as
context. Versus fine-tuning: documents change daily (re-indexing takes seconds; re-training doesn't), answers
can cite their sources, nothing is "baked into" weights (you can delete a document), and it needs no GPU
training. Fine-tuning teaches *style or skills*, retrieval supplies *facts*.

**3. Why both an extractive and a generative mode?**
Extractive answers are copied, so they can't be hallucinated and are cheap — but only short spans.
Generative answers can combine and explain, but can be wrong, so they must cite sources and we measure
refusal/hallucination rates. Offering both lets the user choose verifiability or fluency.

## Retrieval

**4. Bi-encoder vs cross-encoder?**
Bi-encoder encodes query and passage separately → passage vectors precomputed → search is a dot product over
all chunks (fast, scalable) but the model never sees both texts together. Cross-encoder reads them together →
full attention between query and passage words → more accurate, but one forward pass per pair → only feasible
on a shortlist. Hence retrieve (bi-encoder/BM25) → re-rank (cross-encoder).

**5. Explain BM25.**
Sum over query words of IDF × saturating term frequency with length normalisation. IDF makes rare words count
more; k1 controls how quickly repeats stop adding score; b penalises long chunks. I implemented it with Lucene's
`log(1 + …)` IDF because the textbook version goes negative for words in over half the chunks — a test caught it.

**6. Why hybrid search? How do you combine scores on different scales?**
Dense is good at paraphrases, BM25 at exact rare terms (names, codes, numbers). Their scores aren't comparable
(BM25 unbounded, cosine 0–1), so I use Reciprocal Rank Fusion: Σ 1/(60 + rank). It needs no score calibration and
a document ranked well by both wins.

**7. Your results show BM25 beating embeddings. Doesn't that mean embeddings are useless?**
No — it reflects the benchmark: SQuAD questions were written while looking at the paragraph, so they share its
words. On paraphrased real-world questions dense usually wins. I don't have to bet on either: hybrid had the
best deep recall and, after re-ranking, the best rank-1 accuracy. Next step: evaluate on real user queries.

**8. How did you choose the chunk size? What about overlap?**
Experiment: 128/256/384 tokens. Smaller chunks give slightly better rank-1/MRR, larger ones slightly better
deep recall; 256 balances deep recall for the re-ranker, index size and reader input length. Overlap (~1–2
sentences) keeps facts at chunk boundaries together with their context.

**9. How do you know which chunk is "relevant" in your evaluation?**
It's from the question's article and contains a gold answer string — standard answer-recall labelling. It can
mark a chunk relevant that mentions the answer in another context, so it's slightly generous; that's stated.

**10. Recall@k vs MRR?**
Recall@k: is a relevant chunk anywhere in the top k (if not, nothing downstream can answer). MRR: 1/rank of
the first relevant result, averaged — rewards putting it first. I pass 5 chunks on, so Recall@5 and MRR matter.

**11. How would you scale to 10 million chunks?**
Approximate nearest-neighbour index (HNSW or IVF-PQ in FAISS, or a vector DB / pgvector), a real inverted index
for BM25 (OpenSearch/Elasticsearch), sharding, batching query encoding, caching frequent queries, keeping the
cross-encoder on GPU, and only re-ranking the top 30–50.

## Question answering

**12. How does the extractive model find the answer?**
For each token it predicts a start logit and an end logit; best span maximises start[i] + end[j] with i ≤ j and
a max length. Offsets map tokens back to characters for highlighting. Long passages are split into overlapping
windows (stride) and the best span over all windows wins.

**13. How does it decide there's no answer?**
SQuAD 2.0 trains the model to put its "no answer" probability on the [CLS] token: null score = start[CLS] +
end[CLS]. I answer only if best span − null score > margin. The margin is tuned on half of the questions and
reported on the other half, so the number isn't optimistic.

**14. What are EM and F1 in SQuAD?**
After normalisation (lower-case, no punctuation/articles, single spaces): EM = exact match; F1 = word-overlap
F1, best over the gold answers. For unanswerable questions only an empty answer scores 1.

**15. What is the "oracle" setting in your evaluation?**
The reader gets the gold paragraph instead of retrieved chunks — measures the reader alone. Oracle minus
full-system score = the cost of retrieval; it tells you which component to improve.

## Generation

**16. How do you reduce hallucinations?**
Context-only system prompt, numbered sources with mandatory citations, an explicit refusal sentence, greedy
decoding, top-5 re-ranked context; then *measure* it: refusal rate on unanswerable questions, citation rate,
contains-gold rate. The UI always shows sources so users can check. Stricter next step: verify each sentence
against its cited source with an NLI model.

**17. Why a 1.5B model? Why not GPT-4?**
Runs locally on a 6 GB GPU (privacy, no per-token cost, offline). It's weaker at reasoning, which is why the
extractive mode exists and why retrieval quality matters so much. The generator is one class — swapping in a
hosted model is a small change.

## Fine-tuning

**18. How did you fine-tune the NER model? What's label alignment?**
DistilBERT + token-classification head on CoNLL-2003 (BIO tags). Words split into sub-words, so only the first
sub-token keeps the word's label and the others get −100, which cross-entropy ignores. AdamW, LR 5e-5, warm-up
+ linear decay, fp16, 3 epochs, entity-level F1 with seqeval.

**19. Why entity-level F1 and not token accuracy?**
Most tokens are "O"; a model predicting only "O" gets high accuracy. Entity F1 counts an entity correct only if
both its span and type are right.

**20. Why warm-up and a small learning rate?**
The classification head starts random, so early gradients are large and noisy; warm-up avoids wrecking the
pretrained weights in the first steps. A small LR overall avoids catastrophic forgetting.

**21. You trained the classifier on only 30k of 120k articles — why?**
Minutes instead of an hour on a laptop GPU, and pretrained encoders saturate quickly on topic classification.
It's a deliberate cost/benefit choice; I'd test the full set if the last points of accuracy mattered.

## Summaries

**22. How do you summarise documents longer than the model's input?**
Map-reduce: summarise ~900-token sections, then summarise the joined summaries (repeat if needed). Evaluated
with ROUGE vs the LEAD-3 baseline, which is very strong on news.

**23. What does ROUGE not tell you?**
It measures word overlap with one reference, not factual correctness — a fluent summary with a wrong number
can score well. Faithfulness needs other checks (NLI-based, QA-based, or human review).

## Engineering

**24. How does upload work without blocking the API?**
Express (multer) streams the file to disk, creates a "queued" document in MongoDB and returns 202. A background
job queue in the server calls the Python ML service to extract, chunk, embed and index (searchable), then a
second queue calls it to enrich (topic, entities, summary). On restart, interrupted jobs are re-queued. At
scale: BullMQ/Celery + Redis and multiple workers.

**24b. Why MERN plus a Python service, and not one app?**
The models need Python (PyTorch, Transformers, FAISS). Everything else — uploads, records, chat history, jobs,
serving the UI — is classic web-app work where Node/Express and MongoDB are simple and fast. Splitting them
means the ML service can run on a GPU machine and scale on its own, a model crash never loses data, and the
browser only ever talks to one API. The price is one extra network hop and one more service to deploy.

**24c. Why MongoDB? What is embedded and what is referenced?**
The data is document-shaped: a chat session is always read with its messages, so messages are **embedded** in
the session. Pages are a separate collection that **references** the document, because a long PDF has hundreds
of pages and the document list must not load them. Entities are a small array embedded in the document.

**25. What if the FAISS index and the stored chunks disagree?**
The ML service keeps a chunk store (vector id → text, document, pages) next to the FAISS file; the vector id is
its key. At startup, if the vector count doesn't match the store, the index is rebuilt by re-embedding the
stored chunk texts. Both files are written with write-then-rename so a crash never leaves half a file.

**26. How did you test code that depends on large models?**
Model *quality* by evaluation scripts on labelled data; *plumbing* by API tests with tiny fake models (hash
embedder, word-overlap re-ranker) so the full Python flow runs in seconds in CI without downloads. The Node
server is tested against a real MongoDB and a fake ML service (a 40-line Express app), so its tests need no
Python at all.

**27. Tell me about a bug you found.**
Pick from HOW_IT_WAS_BUILT §10. Best: measuring "does each gold answer survive PDF extraction" revealed
non-breaking spaces and soft hyphens silently breaking half the answers — fixed with NFKC normalisation and
vocabulary-based hyphen repair, from 48.8 % to 99.9 %, with regression tests.

**28. What would you do next?**
Evaluate on real user questions; query rewriting for follow-up questions; OCR + layout-aware extraction;
NLI-based faithfulness checks for generated answers; auth and per-user document isolation; ANN index at scale.

## From live testing

**29. Your threshold maximises F1 but hides correct answers. How did you handle that?**
Three outcomes by reader score: above the tuned margin → answer; between 0 and the margin → "not found" plus a
clearly labelled low-confidence suggestion highlighted in its source; ≤ 0 → nothing. Metrics only count real
answers; the suggestion keeps a correct-but-uncertain answer one click away without claiming it.

**30. Show me a failure of your system.**
"Who patented the first steam engine?" → "Thomas Newcomen" with high confidence — wrong (the document says
Ayanz, 1606). Retrieval had the right chunk at rank 2; the reader preferred "invented by Newcomen". The reader
scores passages independently, so scores aren't calibrated across passages. Fixes: joint scoring, a larger
reader, or letting the LLM compare sources. That's why every answer shows its source.

**31. Tell me about a concurrency bug.**
Lazy model loading used one global lock; the worker held it while loading a slow model, so every user question
waited minutes. Fixed with per-model locks plus a regression test. Related: indexing and enrichment shared one
queue, so a slow summary blocked new uploads; split into two queues.

**32. Did prompt engineering fix your LLM's hallucinations?**
No — and that's the interesting part. With instructions only, the 1.5B model invented answers to 87 % of
unanswerable questions. Adding stricter rules and two worked examples improved answer quality and citations
but pushed hallucination to 97 %: it became more eager to answer. What worked was architecture: use the
extractive reader (85 % correct at "no answer") as an answerability gate and skip the LLM when it finds
nothing — hallucination fell to 24 %, 38 % faster, at the cost of refusing 8 % of answerable questions.

**33. Your classifier confuses which classes, and why?**
Business ↔ Sci/Tech (140 and 113 of the errors): tech-company and finance-of-technology news genuinely
overlaps. Sports is nearly perfect — distinctive vocabulary.
