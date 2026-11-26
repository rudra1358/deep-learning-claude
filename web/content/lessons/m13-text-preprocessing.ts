import type { Lesson } from "@/lib/types";

const lesson: Lesson = {
  id: "m13/text-preprocessing",
  title: "Text Preprocessing and Subword Tokenization",
  summary:
    "How raw text becomes integer token IDs: Unicode normalization, word- and character-level tokenization and where they fail, then BPE (worked merge by merge), byte-level BPE, WordPiece, Unigram LM and SentencePiece, choosing a vocabulary size, special tokens, padding, attention masks and truncation.",
  minutes: 70,
  level: "intermediate",
  prerequisites: ["m09/rnn-basics", "m05/classification-losses"],
  next: ["m13/word-embeddings", "m11/tokens-embeddings", "m13/language-modeling"],
  terms: ["token", "bpe", "embedding"],
  sections: {
    // ------------------------------------------------------------------ intro
    intro: [
      {
        t: "p",
        text: "Neural networks work on numbers, not text. **Tokenization** splits text into pieces called **tokens** and gives each an integer **ID** from a fixed list, the **vocabulary**. Each ID picks a row of an embedding matrix, so everything the model learns about language is in terms of these pieces. A model can never see a difference its tokenizer has erased, and it pays in sequence length whenever text is chopped finely.",
      },
      {
        t: "p",
        text: "Early neural NLP used whole words as tokens. Then you choose between a huge vocabulary and many **out-of-vocabulary (OOV)** words that all become one `<unk>` (\"unknown\") token. Characters fix OOV but make sequences several times longer. **Subword tokenization** sits in between: frequent words are single tokens, rare words are spelled from smaller pieces, and nothing is unknown. **Sennrich, Haddow and Birch (2016)**, *Neural Machine Translation of Rare Words with Subword Units*, made it popular by adapting the byte-pair encoding compression algorithm.",
      },
      {
        t: "p",
        text: "Every large language model today uses a subword tokenizer: byte-level BPE (GPT-2 and later), WordPiece (BERT), or Unigram/BPE via SentencePiece (T5, many multilingual models). The choice affects model size, speed, fairness across languages, arithmetic skill and price, since APIs bill per token.",
      },
      {
        t: "callout",
        kind: "key",
        title: "The pipeline",
        text: "raw text → **normalization** (Unicode, case, whitespace) → **pre-tokenization** (split into words / chunks) → **subword model** (BPE, WordPiece, Unigram) → **post-processing** (special tokens) → **batching** (padding, attention masks, truncation) → integer tensor.",
      },
    ],

    // ------------------------------------------------------------------ prerequisites
    prerequisites: [
      {
        t: "list",
        items: [
          "[Sequential data and RNNs](/learn/m09/rnn-basics) — models read a sequence of vectors; tokenization decides what each position stands for.",
          "[Classification losses](/learn/m05/classification-losses) — a language model's output layer is a softmax over the vocabulary, so vocabulary size = number of classes.",
          "Basic probability: frequencies, the probability of independent events as a product, and log-likelihood as a sum (used by WordPiece and Unigram).",
          "Python strings and dictionaries; knowing that a string is a sequence of Unicode code points (numbered characters) and that UTF-8 stores each as 1–4 bytes.",
        ],
      },
    ],

    // ------------------------------------------------------------------ intuition
    intuition: [
      {
        t: "callout",
        kind: "analogy",
        title: "Building words from LEGO",
        text: "A word vocabulary is a box of pre-built models: perfect for those, useless for anything else. A character vocabulary is a box of single studs: you can build anything, but slowly. A subword vocabulary is a well-designed set: common shapes (\"the\", \"ing\", \"tion\") come pre-assembled, and anything unusual is built from smaller bricks.",
      },
      {
        t: "p",
        text: "Byte-pair encoding learns which bricks to pre-assemble by counting. Start with characters. Glue the most common pair of neighbouring symbols into one new symbol, and repeat. After a few thousand merges, frequent words, prefixes and suffixes are single tokens, and rare words fall back to shorter pieces.",
      },
      {
        t: "p",
        text: "Because merges come from counts, a tokenizer mirrors its training text. Trained on English, it splits English efficiently and other languages into many short pieces; trained on code, it learns tokens for runs of spaces and keywords. One sentence can cost very different token counts under different tokenizers.",
      },
      {
        t: "table",
        head: ["Granularity", "\"unhappiness\" becomes", "Vocabulary size", "Unknown words?", "Sequence length"],
        rows: [
          ["Word", "`unhappiness` (or `<unk>`)", "100k–1M+", "Yes, common", "Shortest"],
          ["Subword", "`un` `happi` `ness` (illustrative)", "~30k–250k", "No (with byte fallback)", "Medium"],
          ["Character", "`u n h a p p i n e s s`", "~100s (or thousands for CJK)", "Rarely", "Longest"],
          ["Byte", "11 bytes", "256", "Never", "Longest (more for non-ASCII)"],
        ],
      },
    ],

    // ------------------------------------------------------------------ mechanism
    mechanism: [
      {
        t: "h",
        text: "1. Normalization",
      },
      {
        t: "p",
        text: "Normalization makes text that *should* count as the same into the same string, before any splitting.",
      },
      {
        t: "list",
        items: [
          "**Unicode normalization.** One visible character can be stored in several ways: `é` can be one code point (U+00E9) or `e` plus a combining accent (U+0301). **NFC/NFD** join or split such sequences. **NFKC** also applies *compatibility* mappings: ligatures (`ﬁ` → `fi`), full-width letters (`Ｈ` → `H`), superscripts (`²` → `2`). SentencePiece's default normalizer uses NFKC.",
          "**Lowercasing.** Shrinks the vocabulary and merges `The`/`the`, but loses information (`US` vs `us`, names). BERT has separate cased and uncased models; modern LLMs keep case.",
          "**Accent stripping.** Removing accents (BERT-uncased does this) helps English but hurts languages where accents change meaning.",
          "**Punctuation and whitespace.** Collapse repeated spaces. Either drop punctuation (classic bag-of-words pipelines) or make it separate tokens (most neural tokenizers, because punctuation carries meaning).",
        ],
      },
      {
        t: "code",
        lang: "python",
        title: "Unicode normalization in practice",
        code: String.raw`import unicodedata

samples = ["ﬁne", "Ｈｅｌｌｏ", "x²", "①", "café"]   # ligature, full-width, superscript, circled digit, e + combining accent
for s in samples:
    n = unicodedata.normalize("NFKC", s)
    print(f"{s!r:22} len={len(s)}  ->  NFKC {n!r:10} len={len(n)}")

composed, decomposed = "café", "café"           # look identical on screen
print("equal before normalization:", composed == decomposed)
print("equal after NFKC:          ", unicodedata.normalize("NFKC", composed) == unicodedata.normalize("NFKC", decomposed))
print("UTF-8 bytes of 'é':", list("é".encode("utf-8")), " of '€':", list("€".encode("utf-8")))`,
        output: `'ﬁne'                  len=3  ->  NFKC 'fine'     len=4
'Ｈｅｌｌｏ'                len=5  ->  NFKC 'Hello'    len=5
'x²'                   len=2  ->  NFKC 'x2'       len=2
'①'                    len=1  ->  NFKC '1'        len=1
'café'                len=5  ->  NFKC 'café'     len=4
equal before normalization: False
equal after NFKC:           True
UTF-8 bytes of 'é': [195, 169]  of '€': [226, 130, 172]`,
      },
      {
        t: "h",
        text: "2. Word-level tokenization and the OOV problem",
      },
      {
        t: "p",
        text: "Split on spaces and punctuation, count words in the training text, keep the $V$ most frequent, and map the rest to `<unk>`. Word counts are heavy-tailed (Zipf-like): a few words are very common, and a long tail appears once or twice. Whatever $V$ you pick, new text has words outside it — names, typos, numbers, inflections (`run`, `runs`, `running`, `rerun` get unrelated IDs), German compounds, long Turkish or Finnish words. Every `<unk>` wipes out that word's information, and related words share no parameters.",
      },
      {
        t: "h",
        text: "3. Character-level tokenization",
      },
      {
        t: "p",
        text: "Make each character a token. The vocabulary is tiny and nothing is OOV. But English sequences get about 4–5 times longer, multiplying the cost of RNNs and of quadratic attention, and the model must learn spelling before meaning. Byte-level models go further (256 symbols, no OOV at all), with even longer sequences for non-ASCII scripts: `é` is 2 bytes, `€` is 3.",
      },
      {
        t: "h",
        text: "4. Subword tokenization: shared structure",
      },
      {
        t: "p",
        text: "Every subword method has a **training** phase (build a vocabulary of chosen size from a corpus) and an **encoding** phase (split new text with it). Most first **pre-tokenize** into words, so tokens never cross word boundaries. They differ in how the vocabulary is built:",
      },
      {
        t: "list",
        items: [
          "**BPE** — bottom-up: start from characters, keep merging the most **frequent** neighbouring pair.",
          "**WordPiece** — bottom-up: merge the pair that most raises the **likelihood** of the training data, i.e. pairs that appear together more than their parts' frequencies predict.",
          "**Unigram LM** — top-down: start from a big candidate vocabulary and keep **removing** the pieces whose loss hurts the corpus likelihood least.",
        ],
      },
      {
        t: "h",
        text: "5. Byte-Pair Encoding (BPE)",
      },
      {
        t: "steps",
        items: [
          { title: "Pre-tokenize and count words", text: "Normalize, split into words, count how often each distinct word appears." },
          { title: "Initialise", text: "Split every word into characters and add an end-of-word marker (Sennrich et al. write `</w>`; we write `_`). The marker tells `est` at the end of `newest` apart from `est` inside `establish`. The base vocabulary is the set of distinct symbols." },
          { title: "Count pairs", text: "In each word, count every neighbouring symbol pair, weighted by how often the word appears." },
          { title: "Merge the most frequent pair", text: "Pick the pair with the highest count (with a fixed tie-break rule), add the joined symbol to the vocabulary, record the merge rule, and replace every occurrence in every word." },
          { title: "Repeat", text: "Recount pairs and merge again until the vocabulary reaches its target size (number of merges = target size − base alphabet size)." },
          { title: "Encode new text", text: "Split each new word into characters + marker and apply the learned merges **in the order they were learned**. The result is always the same." },
        ],
      },
      {
        t: "h",
        text: "6. Byte-level BPE (GPT-2)",
      },
      {
        t: "p",
        text: "Character-level BPE still needs a base alphabet of every possible character — over 140,000 Unicode code points — or an `<unk>` for unseen ones. **Byte-level BPE**, introduced with GPT-2 (Radford et al., 2019), runs BPE on the UTF-8 **bytes** instead. The base alphabet is exactly 256 symbols, so any string — emoji, code, any script — encodes with no unknown token. Each byte is shown as a printable character; the space byte appears as `Ġ`, so `Ġcat` means \" cat\" with its leading space. GPT-2 also pre-tokenizes with a regular expression that keeps letters, digits and punctuation apart, so merges cannot glue a word to punctuation. Its vocabulary has 50,257 entries: 256 bytes, 50,000 merges and one end-of-text token.",
      },
      {
        t: "h",
        text: "7. WordPiece (BERT)",
      },
      {
        t: "p",
        text: "WordPiece came from Google's Japanese and Korean voice search (Schuster and Nakajima, 2012), later used in Google's translation system and BERT. Differences from BPE:",
      },
      {
        t: "list",
        items: [
          "**Continuation prefix.** A word's first piece is written plainly; every later piece starts with `##`. `tokenizers` → `token`, `##izers`. So word boundaries are marked on the continuation pieces, not with an end-of-word marker.",
          "**Merge criterion.** Merge the pair with the highest $\\text{score}(a,b) = \\dfrac{\\text{count}(ab)}{\\text{count}(a)\\,\\text{count}(b)}$ instead of the raw count. This merge most raises corpus likelihood under a unigram model. In plain words: it favours pairs frequent *compared with their parts* — pieces that really belong together. Google never published its exact training code; open implementations such as Hugging Face `tokenizers` use this score.",
          "**Encoding.** BERT encodes each word by **greedy longest-match-first**: take the longest vocabulary entry matching the start of the word, then the longest `##` entry matching the rest, and so on. If at some point nothing matches, the **whole word** becomes `[UNK]`.",
        ],
      },
      {
        t: "h",
        text: "8. Unigram language model and SentencePiece",
      },
      {
        t: "p",
        text: String.raw`The **Unigram LM** tokenizer (Kudo, 2018, *Subword Regularization*) treats a split $\mathbf{s} = (s_1,\dots,s_m)$ of a word as independent draws of pieces, with probability $P(\mathbf{s}) = \prod_k p(s_k)$. Training starts from a big seed vocabulary (e.g. frequent substrings) and fits piece probabilities with the EM algorithm. Then it repeatedly removes a share of pieces (e.g. 20%) whose removal lowers corpus log-likelihood least, always keeping single characters so everything stays encodable. Encoding picks the most probable split with the **Viterbi** algorithm. Being a probability model, it can also **sample** other splits during training (subword regularization), a form of data augmentation.`,
      },
      {
        t: "p",
        text: "**SentencePiece** (Kudo and Richardson, 2018) is a library that trains Unigram or BPE directly on raw text, with no language-specific pre-tokenization. It first replaces spaces with the visible symbol `▁` (U+2581): `Hello world` becomes `▁Hello▁world`, and spaces become ordinary characters that can sit inside tokens. Decoding is lossless: join the pieces and turn `▁` back into spaces. So it works for languages without spaces (Japanese, Chinese, Thai). T5 and many multilingual models use SentencePiece vocabularies.",
      },
      {
        t: "h",
        text: "9. Special tokens, padding, attention masks and truncation",
      },
      {
        t: "table",
        head: ["Token", "Used by", "Purpose"],
        rows: [
          ["`[CLS]`", "BERT", "Added at the start of every input; its final hidden state represents the whole sequence for classification."],
          ["`[SEP]`", "BERT", "Separates and ends segments: `[CLS] A [SEP] B [SEP]` for sentence pairs."],
          ["`[PAD]` / `<pad>`", "all", "Fills short sequences up to the batch length; ignored through the attention mask and the loss."],
          ["`[UNK]` / `<unk>`", "word-level, WordPiece", "Stands for anything not in the vocabulary (byte-level BPE never needs it)."],
          ["`[MASK]`", "BERT", "Replaces tokens to be predicted in masked language modelling."],
          ["`<bos>` / `<s>`", "many decoders", "Beginning of sequence: gives the first prediction some context."],
          ["`<eos>` / `</s>` / `<|endoftext|>`", "decoders, seq2seq", "End of sequence: tells a generator when to stop, and separates documents."],
        ],
      },
      {
        t: "list",
        items: [
          "**Padding.** Tensors must be rectangular, so each sequence in a batch is filled with the pad ID up to the longest (or a fixed length). Encoders usually pad on the right; decoder-only models generating in batches often pad on the left, so the last position is real text.",
          "**Attention mask.** A 0/1 array of the same shape: 1 for real tokens, 0 for padding. Attention layers add $-\\infty$ (in practice a large negative number) to scores at masked spots, so padding gets zero attention. Losses skip padded targets (e.g. PyTorch's `ignore_index=-100`).",
          "**Truncation.** Models have a maximum length (512 for BERT). Longer inputs are cut, usually keeping the start; for sentence pairs you can trim the longer segment first. When nothing may be lost (e.g. question answering), long documents are split into overlapping windows (with a *stride*).",
        ],
      },
    ],

    // ------------------------------------------------------------------ math
    math: [
      {
        t: "h",
        text: "Notation",
      },
      {
        t: "list",
        items: [
          "$\\mathcal{D}$ — training corpus, seen as a bag of words $w$ with frequencies $n_w$.",
          "$\\text{seg}(w) = (s_1, \\dots, s_m)$ — how word $w$ is currently split into symbols.",
          "$\\text{count}(a,b) = \\sum_w n_w \\cdot \\#\\{k : s_k = a,\\, s_{k+1} = b\\}$ — how often $a$ is directly followed by $b$, weighted by word frequency.",
          "$\\text{count}(a) = \\sum_w n_w \\cdot \\#\\{k : s_k = a\\}$ — weighted count of symbol $a$.",
          "$V$ — vocabulary size; $|A|$ — size of the base alphabet; $M$ — number of merges, $V = |A| + M$ (+ special tokens).",
          "$p(s)$ — probability of piece $s$ under a Unigram LM, with $\\sum_{s \\in V} p(s) = 1$.",
        ],
      },
      {
        t: "math",
        tex: String.raw`\text{BPE:}\;(a^\ast,b^\ast) = \arg\max_{(a,b)}\,\text{count}(a,b)
\qquad
\text{WordPiece:}\;(a^\ast,b^\ast) = \arg\max_{(a,b)}\,\frac{\text{count}(a,b)}{\text{count}(a)\,\text{count}(b)}`,
        caption: "The two bottom-up merge rules: BPE takes the most frequent pair; WordPiece takes the pair most frequent relative to its parts.",
      },
      {
        t: "math",
        tex: String.raw`\text{Unigram:}\quad \mathbf{s}^\ast(w) = \arg\max_{\mathbf{s} \in S(w)} \sum_{k} \log p(s_k),
\qquad \mathcal{L} = \sum_{w} n_w \log \sum_{\mathbf{s} \in S(w)} \prod_k p(s_k)`,
        caption: "S(w) is the set of all ways to split w into vocabulary pieces. Training maximizes the corpus log-likelihood ℒ; encoding uses the single best (Viterbi) split.",
      },
      {
        t: "h",
        text: "Worked example: BPE merge by merge",
      },
      {
        t: "p",
        text: "Corpus (Sennrich et al.'s example): `low` ×5, `lower` ×2, `newest` ×6, `widest` ×3. Starting split, with `_` as end-of-word marker: `l o w _` (5), `l o w e r _` (2), `n e w e s t _` (6), `w i d e s t _` (3). Base alphabet: `_ d e i l n o r s t w`, 11 symbols. Ties go to the alphabetically smallest pair.",
      },
      {
        t: "p",
        text: "First pair counts: `e s` = 6 (newest) + 3 (widest) = **9**; `s t` = 9; `t _` = 9; `w e` = 6 (newest) + 2 (lower) = 8; `l o` = 5 + 2 = 7; `o w` = 7. Three pairs tie at 9; the tie-break picks `e s`.",
      },
      {
        t: "table",
        head: ["Merge", "Pair (count)", "New symbol", "Split after the merge"],
        rows: [
          ["1", "`e` + `s` (9)", "`es`", "`l o w _` · `l o w e r _` · `n e w es t _` · `w i d es t _`"],
          ["2", "`es` + `t` (9)", "`est`", "`n e w est _` · `w i d est _`"],
          ["3", "`est` + `_` (9)", "`est_`", "`n e w est_` · `w i d est_`"],
          ["4", "`l` + `o` (7)", "`lo`", "`lo w _` · `lo w e r _`"],
          ["5", "`lo` + `w` (7)", "`low`", "`low _` · `low e r _`"],
          ["6", "`e` + `w` (6)", "`ew`", "`n ew est_`"],
          ["7", "`ew` + `est_` (6)", "`ewest_`", "`n ewest_`"],
          ["8", "`n` + `ewest_` (6)", "`newest_`", "`newest_`"],
          ["9", "`low` + `_` (5)", "`low_`", "`low_` · `low e r _`"],
          ["10", "`d` + `est_` (3)", "`dest_`", "`w i dest_`"],
        ],
      },
      {
        t: "p",
        text: "After 10 merges the vocabulary has $11 + 10 = 21$ symbols, and the four words cost 1 + 4 + 1 + 3 = 9 tokens instead of 4 + 6 + 7 + 7 = 24 starting symbols (per occurrence). Merges 1–3 built the suffix `est_` — a real English word ending, found purely by counting.",
      },
      {
        t: "p",
        text: "**Encoding an unseen word.** `lowest` → `l o w e s t _`. Apply merges in order: 1 → `l o w es t _`; 2 → `l o w est _`; 3 → `l o w est_`; 4 → `lo w est_`; 5 → `low est_`; merges 6–10 do not apply. Result: **`low` `est_`** — two meaningful pieces for a word never seen in training. The table and this encoding come from the Python code below and from the simulator's TypeScript code, which agree.",
      },
      {
        t: "h",
        text: "Worked example: WordPiece scores on the same corpus",
      },
      {
        t: "p",
        text: "With WordPiece's `##` style, the words start as `l ##o ##w`, `l ##o ##w ##e ##r`, `n ##e ##w ##e ##s ##t`, `w ##i ##d ##e ##s ##t`. Symbol counts include `##e` = 2 + 12 + 3 = 17 and `##s` = 9. Scores:",
      },
      {
        t: "table",
        head: ["Pair", "count(ab)", "count(a)", "count(b)", "score"],
        rows: [
          ["`w` `##i`", "3", "3", "3", "3/9 = 0.3333"],
          ["`##i` `##d`", "3", "3", "3", "0.3333"],
          ["`l` `##o`", "7", "7", "7", "7/49 = 0.1429"],
          ["`##s` `##t`", "9", "9", "9", "9/81 = 0.1111"],
          ["`##e` `##s`", "9", "17", "9", "9/153 = 0.0588"],
        ],
      },
      {
        t: "p",
        text: "BPE would merge `##e ##s` (the most frequent pair, count 9). WordPiece ranks it last here because `##e` is common everywhere. It prefers `w ##i`, which is rare but whose parts *always* appear together. That is the likelihood rule at work.",
      },
      {
        t: "h",
        text: "Worked example: Unigram Viterbi segmentation",
      },
      {
        t: "p",
        text: "Suppose a Unigram vocabulary (part of a larger one whose probabilities sum to 1) contains `low` 0.10, `est` 0.08, `lo` 0.03, `west` 0.02, `es` 0.05, `we` 0.02, `st` 0.03, and the single letters `l` 0.04, `o` 0.04, `w` 0.04, `e` 0.06, `s` 0.04, `t` 0.05. Possible splits of `lowest`:",
      },
      {
        t: "table",
        head: ["Split", "Probability"],
        rows: [
          ["`low` `est`", "0.10 × 0.08 = 0.008"],
          ["`lo` `west`", "0.03 × 0.02 = 0.0006"],
          ["`low` `es` `t`", "0.10 × 0.05 × 0.05 = 0.00025"],
          ["`lo` `we` `st`", "0.03 × 0.02 × 0.03 = 0.000018"],
          ["`l` `o` `w` `e` `s` `t`", "≈ 7.7 × 10⁻⁹"],
        ],
      },
      {
        t: "p",
        text: "Viterbi returns `low` `est` with log-probability $\\ln 0.008 = -4.828$. Each extra piece multiplies in another probability below 1, so Unigram naturally prefers few, frequent pieces. Subword regularization instead samples splits in proportion to (a power of) these probabilities.",
      },
      {
        t: "h",
        text: "Vocabulary size in numbers",
      },
      {
        t: "p",
        text: String.raw`The input embedding matrix has $V \times d$ parameters, and so does the output softmax layer (unless the two are tied, i.e. shared). With $d = 768$: $V = 32{,}000$ costs $24.6$M parameters per matrix; $V = 256{,}000$ costs $196.6$M. In return, a bigger vocabulary gives shorter sequences: less compute per document (linearly for RNNs, quadratically for Transformer attention) and more text per context window. But rare tokens in a huge vocabulary get few gradient updates, so their embeddings are poorly trained.`,
      },
    ],

    // ------------------------------------------------------------------ visual
    visual: [
      {
        t: "sim",
        id: "tokenizer",
        caption: "BPE trained live on the text in the box. Step through merges one at a time; the right panel encodes any new text with the merges learned so far.",
      },
      {
        t: "list",
        ordered: true,
        items: [
          "At step 0 every word is a row of single-letter chips. Read the pair-count table: the top pair is the next merge. Press **Step** and check that every occurrence of that pair turns into one gold chip.",
          "Keep stepping and watch **chars / token** rise: each merge shortens the corpus in tokens. Frequent words (`the_`, `sat_`, `on_`) become single tokens within the first dozen merges.",
          "Look for merges that build endings such as `s_` and `st_` — word structure appearing from counts alone.",
          "Type `the newest dogs sat on the catalogs` in the encode box. `catalogs` is not in the corpus, yet it is built from known pieces. Then type a word with letters missing from the corpus (e.g. `zebra`): unknown letters become `<unk>` (id 0) — exactly the problem byte-level BPE removes.",
          "Swap the corpus for a paragraph in another language or some source code, and compare which merges come first.",
          "Set **Merges to learn** to 5, then 80: training stops early when no pair appears at least twice, and the narration says so.",
        ],
      },
    ],

    // ------------------------------------------------------------------ scratch
    scratch: [
      {
        t: "p",
        text: "A complete BPE trainer and encoder in pure Python (standard library only; NumPy does not help with string processing). It follows the steps above exactly, including the fixed tie-break, and learns the same merges as the simulator.",
      },
      {
        t: "code",
        lang: "python",
        title: "bpe_from_scratch.py",
        code: String.raw`import re
import unicodedata
from collections import Counter

EOW = "_"   # end-of-word marker (Sennrich et al. write it as </w>)

def normalize(text):
    """NFKC-normalize, lowercase, and replace anything that is not a letter, digit or space by a space."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^a-z0-9\s]", " ", text)

def pretokenize(text):
    """Split normalized text into words; BPE never merges across word boundaries."""
    return normalize(text).split()

def count_pairs(word_freqs, segs):
    pairs = Counter()
    for w, f in word_freqs.items():
        s = segs[w]
        for a, b in zip(s, s[1:]):
            pairs[(a, b)] += f          # each occurrence weighted by word frequency
    return pairs

def apply_merge(symbols, a, b):
    """Replace every non-overlapping occurrence of the adjacent pair (a, b), left to right."""
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
            out.append(a + b); i += 2
        else:
            out.append(symbols[i]); i += 1
    return out

def train_bpe(text, num_merges):
    word_freqs = Counter(pretokenize(text))
    segs = {w: list(w) + [EOW] for w in word_freqs}
    vocab = sorted({ch for s in segs.values() for ch in s})       # base alphabet
    merges = []
    for _ in range(num_merges):
        pairs = count_pairs(word_freqs, segs)
        if not pairs:
            break
        # most frequent pair; ties broken lexicographically so results are reproducible
        (a, b), n = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))
        if n < 2:
            break                       # merging a pair seen once compresses nothing useful
        merges.append((a, b))
        vocab.append(a + b)
        segs = {w: apply_merge(s, a, b) for w, s in segs.items()}
    return merges, vocab

def encode(text, merges, token_to_id):
    """Apply learned merges, in the order learned, to each word of new text."""
    ids, pieces = [], []
    for w in pretokenize(text):
        s = list(w) + [EOW]
        for a, b in merges:
            s = apply_merge(s, a, b)
        for tok in s:
            pieces.append(tok)
            ids.append(token_to_id.get(tok, token_to_id["<unk>"]))
    return pieces, ids

corpus = """The cat sat on the mat. The cats sat on the mats.
A dog sat on the log; the dogs sat on the logs.
Newer models train faster, and the newest models train fastest."""

merges, vocab = train_bpe(corpus, num_merges=30)
vocab = ["<unk>"] + vocab
token_to_id = {t: i for i, t in enumerate(vocab)}

print("number of merges learned:", len(merges))
print("first 12 merges:", [a + "+" + b for a, b in merges[:12]])
print("vocabulary size:", len(vocab))

for sentence in ["The cats sat on the mats.", "the fastest dogs", "Lowest catalogs!", "Zebra"]:
    pieces, ids = encode(sentence, merges, token_to_id)
    print(f"\n{sentence!r}")
    print("  pieces:", pieces)
    print("  ids:   ", ids)
    print(f"  {len(normalize(sentence).replace(' ', ''))} non-space characters -> {len(ids)} tokens")`,
        output: `number of merges learned: 30
first 12 merges: ['a+t', 'e+_', 'h+e_', 't+he_', 'at+_', 'n+_', 's+_', 'o+g', 'o+n_', 's+at_', 's+t', 'a+i']
vocabulary size: 48

'The cats sat on the mats.'
  pieces: ['the_', 'c', 'ats_', 'sat_', 'on_', 'the_', 'm', 'ats_']
  ids:    [21, 3, 33, 27, 26, 21, 11, 33]
  19 non-space characters -> 8 tokens

'the fastest dogs'
  pieces: ['the_', 'faste', 'st_', 'dog', 's_']
  ids:    [21, 40, 47, 35, 24]
  14 non-space characters -> 5 tokens

'Lowest catalogs!'
  pieces: ['l', 'o', 'w', 'e', 'st_', 'c', 'at', 'a', 'log', 's_']
  ids:    [10, 13, 17, 5, 47, 3, 18, 2, 41, 24]
  14 non-space characters -> 10 tokens

'Zebra'
  pieces: ['z', 'e', 'b', 'r', 'a', '_']
  ids:    [0, 5, 0, 14, 2, 1]
  5 non-space characters -> 6 tokens`,
      },
      {
        t: "list",
        items: [
          "Vocabulary size 48 = `<unk>` + 17 base symbols + 30 merges.",
          "`the fastest` splits as `faste` + `st_`. The merges came from `faster` and `fastest`, and the greedy merge order, not grammar, decides the boundary. Real tokenizers make the same kind of odd splits.",
          "`lowest` is spelled letter by letter except `st_`, because this corpus never contained `low`. Tokenization quality depends entirely on the training text.",
          "`z` and `b` never appear in the corpus, so they map to `<unk>` (id 0). Byte-level BPE avoids this by starting from all 256 bytes.",
          "This code recounts all pairs after every merge, costing $O(M \\cdot N)$ for $M$ merges over $N$ symbols. Production trainers update only the pair counts next to each merge, using a priority queue.",
        ],
      },
    ],

    // ------------------------------------------------------------------ framework
    framework: [
      {
        t: "tabs",
        tabs: [
          {
            label: "Hugging Face tokenizers",
            blocks: [
              {
                t: "p",
                text: "The `tokenizers` library (Rust core, Python bindings) implements BPE, WordPiece and Unigram with configurable normalizers, pre-tokenizers and post-processors. We train a byte-level BPE (GPT-2 recipe) and a WordPiece tokenizer (BERT recipe) on a tiny corpus, then encode a batch with `[CLS]`/`[SEP]`, padding, truncation and attention masks. Install with `pip install tokenizers` (tested with version 0.23.3). In PyTorch, pass the `ids` and `attention_mask` lists straight to `torch.tensor`.",
              },
              {
                t: "code",
                lang: "python",
                title: "hf_tokenizers_demo.py (WordPiece IDs vary between runs)",
                code: String.raw`from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders, processors

corpus = [
    "The cat sat on the mat. The cats sat on the mats.",
    "A dog sat on the log; the dogs sat on the logs.",
    "Newer models train faster, and the newest models train fastest.",
    "Tokenizers split text into subword units before embedding.",
] * 50   # repeat so pair counts are large enough to merge

# ---- 1. Byte-level BPE (the GPT-2 recipe) ----
bpe = Tokenizer(models.BPE())
bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe.decoder = decoders.ByteLevel()
bpe.train_from_iterator(corpus, trainers.BpeTrainer(
    vocab_size=350, special_tokens=["<pad>", "<bos>", "<eos>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))   # all 256 bytes -> no <unk> ever
enc = bpe.encode("The newest cats sat. Zebra ü!")
print("byte-level BPE vocab size:", bpe.get_vocab_size())
print("tokens:", enc.tokens)
print("decoded:", bpe.decode(enc.ids))

# ---- 2. WordPiece (the BERT recipe) with [CLS]/[SEP], padding and truncation ----
wp = Tokenizer(models.WordPiece(unk_token="[UNK]"))
wp.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
wp.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
wp.train_from_iterator(corpus, trainers.WordPieceTrainer(
    vocab_size=120, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
wp.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
    special_tokens=[("[CLS]", wp.token_to_id("[CLS]")), ("[SEP]", wp.token_to_id("[SEP]"))])
wp.enable_padding(pad_id=wp.token_to_id("[PAD]"), pad_token="[PAD]")
wp.enable_truncation(max_length=8)

batch = wp.encode_batch(["The cats sat today.", "Newest tokenizers train fastest on logs today"])
for e in batch:
    print("\ntokens:        ", e.tokens)
    print("ids:           ", e.ids)
    print("attention_mask:", e.attention_mask)`,
                output: `byte-level BPE vocab size: 348
tokens: ['The', 'Ġnewest', 'Ġcats', 'Ġsat', '.', 'Ġ', 'Z', 'e', 'b', 'r', 'a', 'Ġ', 'Ã', '¼', '!']
decoded: The newest cats sat. Zebra ü!

tokens:         ['[CLS]', 'the', 'cats', 'sat', '[UNK]', '.', '[SEP]', '[PAD]']
ids:            [2, 52, 100, 54, 1, 5, 3, 0]
attention_mask: [1, 1, 1, 1, 1, 1, 1, 0]

tokens:         ['[CLS]', 'newest', 'tokeni', '##zers', 'train', 'fastest', 'on', '[SEP]']
ids:            [2, 108, 115, 119, 76, 106, 53, 3]
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1]`,
              },
              {
                t: "callout",
                kind: "note",
                title: "Reading the output",
                text: "This output is from a real run. On this tiny, repetitive corpus many merges tie exactly, and the WordPiece trainer breaks ties in no fixed order, so repeated runs can split `tokenizers` differently (e.g. `token` + `##izers`) and give different IDs. The output's structure stays the same.",
              },
              {
                t: "list",
                items: [
                  "**Byte-level BPE**: `Ġ` marks a leading space. `ü` is not in the training text, so it becomes its two UTF-8 bytes (195, 188), shown as `Ã` and `¼`; decoding restores the exact string. There is no `<unk>`.",
                  "The trainer stopped at 348 entries (below the requested 350) because no further pair appeared often enough to merge.",
                  "**WordPiece**: `today` contains `y`, which never appears in the corpus. No piece sequence covers the word, so the **whole word** becomes `[UNK]` — not just the letter.",
                  "**Padding and mask**: the first sequence (7 tokens) is padded to the batch length 8 with `[PAD]` (id 0), and its attention mask has a 0 there.",
                  "**Truncation**: the second sequence would be 10 tokens; with `max_length=8` it is cut to 8. `[SEP]` still ends it, because truncation leaves room for the special tokens the post-processor adds. The words `logs today` are lost.",
                ],
              },
            ],
          },
          {
            label: "PyTorch",
            blocks: [
              {
                t: "p",
                text: "PyTorch has no tokenizer of its own: a tokenizer (as in the previous tab, or `transformers.AutoTokenizer` for a pretrained model) makes ID lists, which are padded into tensors. This program pads ID lists with `pad_sequence`, builds the attention mask, and shows a loss with `ignore_index` skipping padded positions, using the IDs from the from-scratch BPE above.",
              },
              {
                t: "code",
                lang: "python",
                title: "padding_and_masks.py",
                code: String.raw`import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

PAD_ID = 48                                         # new id after the 48-entry BPE vocabulary (id 0 is <unk>)
VOCAB = 49
batch_ids = [[21, 3, 33, 27, 26, 21, 11, 33],     # "the cats sat on the mats"
             [21, 40, 47, 35, 24]]                # "the fastest dogs"
seqs = [torch.tensor(s) for s in batch_ids]
ids = pad_sequence(seqs, batch_first=True, padding_value=PAD_ID)   # (B, T_max)
mask = (ids != PAD_ID).long()                                      # 1 = real token, 0 = padding
print("ids:\n", ids)
print("attention_mask:\n", mask)

# Embedding lookup: padding_idx keeps the pad vector at zero and out of training
emb = nn.Embedding(num_embeddings=VOCAB, embedding_dim=4, padding_idx=PAD_ID)
print("embedded shape:", tuple(emb(ids).shape))

# Loss that ignores padded targets
torch.manual_seed(0)
logits = torch.randn(2, ids.shape[1], VOCAB)                         # fake model output (B, T, V)
targets = ids.masked_fill(mask == 0, -100)                         # -100 = ignore
loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1), ignore_index=-100)
print("tokens counted in loss:", int(mask.sum()), "| loss:", round(loss.item(), 4))`,
                output: `ids:
 tensor([[21,  3, 33, 27, 26, 21, 11, 33],
        [21, 40, 47, 35, 24, 48, 48, 48]])
attention_mask:
 tensor([[1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0]])
embedded shape: (2, 8, 4)
tokens counted in loss: 13 | loss: 4.66`,
              },
            ],
          },
          {
            label: "TensorFlow / Keras",
            blocks: [
              {
                t: "p",
                text: "Keras' `TextVectorization` layer does normalization (lowercase, strip punctuation by default), whitespace splitting, vocabulary building with `adapt`, OOV mapping and padding inside the model graph. It is word-level (or character-level with `split=\"character\"`); for subwords in Keras, use KerasNLP / KerasHub tokenizers or a SentencePiece model.",
              },
              {
                t: "code",
                lang: "python",
                title: "keras_text_vectorization.py (not executed in CI)",
                code: String.raw`import tensorflow as tf

corpus = tf.constant([
    "The cat sat on the mat.",
    "The cats sat on the mats.",
    "A dog sat on the log.",
])

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=20,                       # vocabulary size, including padding (0) and OOV [UNK] (1)
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="int",
    output_sequence_length=8,            # pad (with 0) or truncate every sequence to 8
)
vectorizer.adapt(corpus)                 # builds the vocabulary from the corpus, most frequent first

print(vectorizer.get_vocabulary())
print(vectorizer(tf.constant(["The zebra sat on the mat!"])).numpy())`,
                output: "Prints the learned vocabulary, starting with '' (padding, id 0) and '[UNK]' (OOV, id 1), followed by words sorted by frequency ('the', 'sat', 'on', …). Then prints one row of 8 integer IDs for the new sentence: 'zebra' maps to 1 ([UNK]) because it was never seen, and the row ends with 0s for padding.",
              },
            ],
          },
        ],
      },
    ],

    // ------------------------------------------------------------------ variations
    variations: [
      {
        t: "table",
        head: ["Method", "How the vocabulary is built", "Encoding", "Pros", "Cons", "Used by (examples)"],
        rows: [
          ["Word-level", "The $V$ most frequent words", "Look up, else `<unk>`", "Short sequences; easy to read", "OOV; huge vocab; `run`/`runs` share nothing", "Classic RNN/word2vec pipelines, Keras `TextVectorization`"],
          ["Character / byte", "Fixed alphabet", "Split", "No OOV (bytes); tiny vocab", "Very long sequences; model must learn spelling", "Character-level LMs; byte-level models such as ByT5"],
          ["BPE", "Bottom-up merges by pair **frequency**", "Apply merges in order", "Simple, fast, same result every time", "Greedy; splits don't always follow word structure", "Original subword NMT (Sennrich et al.)"],
          ["Byte-level BPE", "BPE over UTF-8 bytes", "Apply merges in order", "No `<unk>` ever; any script or emoji", "Non-Latin scripts cost more tokens", "GPT-2 and later GPT models, RoBERTa"],
          ["WordPiece", "Bottom-up merges by **likelihood** score; `##` continuations", "Greedy longest-match-first", "Favours pieces that belong together", "Unknown character ⇒ whole word `[UNK]`", "BERT, DistilBERT"],
          ["Unigram LM", "Prune a large seed vocab top-down by likelihood (EM)", "Viterbi (or sampling)", "Probabilistic; allows subword regularization", "Slower training; more complex", "T5, ALBERT, XLNet (via SentencePiece)"],
          ["SentencePiece", "Library: BPE or Unigram on raw text, `▁` for spaces", "Lossless", "Any language, no pre-tokenizer, reversible", "Needs its own model file", "T5, mT5, many multilingual models, Llama 2"],
        ],
      },
      {
        t: "p",
        text: "Related ideas: **BPE-dropout** (Provilkov et al., 2020) randomly skips merges during training so the model sees several splits of a word, like Unigram's subword regularization. **Byte fallback** in SentencePiece encodes unknown characters as byte tokens instead of `<unk>`, combining a character vocabulary with byte-level coverage.",
      },
    ],

    // ------------------------------------------------------------------ applications
    applications: [
      {
        t: "list",
        items: [
          "**Large language models**: every GPT-style model reads and writes byte-level BPE tokens; GPT-2's vocabulary has 50,257 entries. Context windows and API prices are counted in these tokens.",
          "**BERT-family encoders**: BERT-base uncased uses a 30,522-entry WordPiece vocabulary with `[CLS]`, `[SEP]`, `[MASK]` and `[PAD]`, and every fine-tuned classifier or QA model inherits it.",
          "**Machine translation**: subword vocabularies, often shared by source and target, solved rare-word translation and let models copy names and numbers.",
          "**Multilingual models**: SentencePiece trained on many languages lets one model handle scripts with and without spaces.",
          "**Code models**: tokenizers trained on source code learn tokens for indentation, common names and operators, making programs much shorter.",
          "**Search and classical NLP**: normalization (NFKC, case folding) and word-level vocabularies are still standard in search engines and bag-of-words or TF-IDF features.",
        ],
      },
    ],

    // ------------------------------------------------------------------ tradeoffs
    tradeoffs: [
      {
        t: "table",
        head: ["Choice", "Gain", "Cost"],
        rows: [
          ["Larger vocabulary", "Shorter sequences; more text per context window; less compute per document", "Bigger embedding and softmax matrices ($V \\times d$ each); rare tokens poorly trained"],
          ["Smaller vocabulary", "Small embedding tables; every token well trained", "Longer sequences; more compute; model must combine more pieces"],
          ["Lowercasing / accent stripping", "Smaller vocab; merges variants", "Loses information (names, accents that carry meaning)"],
          ["Byte-level base", "No unknown tokens, any input", "Non-ASCII text costs 2–4 bytes per character before merges"],
          ["Pre-tokenizing on whitespace", "Tokens line up with words; faster training", "Fails for languages without spaces; not perfectly reversible"],
        ],
      },
      {
        t: "list",
        items: [
          "**Fairness across languages**: a mostly-English tokenizer spends many more tokens per sentence on under-represented languages, making them slower and pricier, with less room in the context window.",
          "**Arithmetic and spelling**: numbers split into uneven chunks (`1234` may become `12` `34` or `123` `4`), making digit-by-digit reasoning harder; some tokenizers split numbers into single digits for this reason. Models also struggle to count letters inside a token they never see spelled out.",
          "**The tokenizer is locked to the model**: changing it after pretraining breaks the embedding matrix, so the choice is effectively permanent.",
        ],
      },
    ],

    // ------------------------------------------------------------------ mistakes
    mistakes: [
      {
        t: "list",
        items: [
          "**Using a different tokenizer at inference than in training.** The IDs point to the wrong embedding rows and the model outputs garbage, often with no error. Fix: always load the tokenizer saved with the model checkpoint.",
          "**Forgetting the attention mask.** Real tokens then attend to padding, and results change with what else is in the batch. Fix: pass `attention_mask` and use `ignore_index` (or a mask) in the loss.",
          "**Applying merges in frequency order instead of learned order when encoding.** BPE encoding must replay merges in the order they were learned; any other order gives splits the model never saw.",
          "**Merging across word boundaries in a homemade BPE.** Without pre-tokenization (or `▁`/end-of-word markers), pairs like `e_t` from `the time` get merged and tokens stop being reusable. Fix: count pairs only within words.",
          "**Normalizing inconsistently.** Training text in NFC and inference text in NFD (common in text copied from macOS file names) give different tokens for identical-looking strings. Fix: use the same normalizer everywhere — it is part of the tokenizer.",
          "**Assuming one token = one word.** English usually has more tokens than words (many other languages far more). This affects context limits, cost estimates and per-token metrics: perplexities from different tokenizers cannot be compared.",
          "**Truncating silently.** Long inputs get cut without warning, dropping the answer in QA or the conclusion of a review. Fix: check lengths, or use a sliding window with stride.",
          "**Padding on the wrong side for batched generation.** Right-padding a decoder-only model puts pad tokens between the prompt and the generated text. Fix: left-pad prompts for generation.",
        ],
      },
    ],

    // ------------------------------------------------------------------ interview
    interview: [
      { t: "qa", level: "beginner", q: "What is the out-of-vocabulary problem and how do subword tokenizers solve it?", a: "With a fixed word vocabulary, any missing word (names, typos, new or inflected forms) becomes one `<unk>` and its content is lost. Subword tokenizers keep all base characters (or all 256 bytes) in the vocabulary, so any word can be spelled from known pieces, while frequent words stay single tokens." },
      { t: "qa", level: "beginner", q: "Describe the BPE training algorithm.", a: "Split each corpus word into characters plus an end-of-word marker. Count all neighbouring symbol pairs, weighted by word frequency. Merge the most frequent pair into a new symbol, add it to the vocabulary, record the merge, and replace it everywhere. Repeat until the vocabulary reaches the target size. To encode new text, replay the recorded merges in order." },
      { t: "qa", level: "beginner", q: "What are padding and attention masks for?", a: "Batches need rectangular tensors, so shorter sequences are filled with a pad token to a common length. The attention mask marks real tokens with 1 and padding with 0, so attention gives padding zero weight and the loss can skip padded positions." },
      { t: "qa", level: "intermediate", q: "How does WordPiece differ from BPE?", a: "Training: BPE merges the most frequent pair; WordPiece merges the pair with the highest $\\text{count}(ab)/(\\text{count}(a)\\text{count}(b))$ — the largest likelihood gain, favouring pairs whose parts rarely appear apart. Representation: WordPiece marks non-initial pieces with `##` instead of an end-of-word marker. Encoding: BERT's WordPiece uses greedy longest-match-first per word, and if any part fails to match the whole word becomes `[UNK]`; BPE replays merges in order." },
      { t: "qa", level: "intermediate", q: "Why did GPT-2 use byte-level BPE?", a: "A character base alphabet must either include every Unicode character (over 140k) or fall back to `<unk>`. Working on UTF-8 bytes makes the base alphabet exactly 256 symbols, guarantees any string can be encoded and decoded losslessly, and keeps the vocabulary at a manageable 50,257 entries." },
      { t: "qa", level: "intermediate", q: "What does SentencePiece's `▁` symbol do?", a: "SentencePiece treats input as a raw character stream and replaces spaces with `▁` (U+2581) before training. Spaces become ordinary characters that can be part of pieces (e.g. `▁the`), no language-specific pre-tokenizer is needed (useful for Japanese or Chinese), and detokenization is exact: join pieces and turn `▁` back into spaces." },
      { t: "qa", level: "intermediate", q: "How do you choose a vocabulary size?", a: "Balance embedding/softmax parameters ($2Vd$ if untied) against sequence length. Larger $V$ shortens sequences (less compute, more text per context window) but adds parameters and undertrains rare tokens; smaller $V$ does the reverse. Typical: 30k–50k for one language, 100k–250k for multilingual. In practice, measure tokens per word (fertility) on target-domain text for a few sizes." },
      { t: "qa", level: "advanced", q: "Explain the Unigram LM tokenizer and subword regularization.", a: "The Unigram model gives each piece a probability and scores a split as the product of its pieces' probabilities. Training starts from a big seed vocabulary, estimates probabilities with EM (expected piece counts over all splits, via a forward–backward pass), and repeatedly prunes the pieces whose removal lowers corpus likelihood least, keeping all characters. Encoding uses Viterbi for the most likely split. Subword regularization (Kudo, 2018) samples splits during training, so the downstream model sees several splits of a word — improving robustness, especially for low-resource translation." },
      { t: "qa", level: "advanced", q: "Why do tokenization choices affect arithmetic and multilingual performance?", a: "Uneven multi-digit chunks hide place value: the same digits can split differently in different contexts, so the model cannot apply one digit-by-digit method; single-digit splitting helps. In multilingual text, merges learned mostly from high-resource languages break other languages into many short pieces, raising cost, shortening effective context and making the task harder. Tokenizer training data should match the intended language and domain mix." },
    ],

    // ------------------------------------------------------------------ practice
    practice: [
      {
        t: "quiz",
        questions: [
          {
            q: "Which rule does BPE use to choose the next merge?",
            options: ["The pair with the highest count(ab) / (count(a)·count(b))", "The most frequent adjacent pair", "The pair whose removal hurts likelihood least", "The longest pair"],
            answer: 1,
            explain: "BPE merges the most frequent neighbouring pair. The ratio is WordPiece; removal-by-likelihood is Unigram.",
          },
          {
            kind: "numeric",
            q: "A BPE tokenizer starts from a base alphabet of 11 symbols and performs 10 merges. How many symbols are in its vocabulary (no special tokens)?",
            answer: 21,
            tolerance: 0,
            explain: "Each merge adds exactly one symbol: 11 + 10 = 21.",
          },
          {
            kind: "numeric",
            q: "Corpus: `newest` ×6 and `widest` ×3, split into characters. What is the frequency-weighted count of the pair (e, s)?",
            answer: 9,
            tolerance: 0,
            explain: "`es` appears once in each word: 6 + 3 = 9.",
          },
          {
            q: "In GPT-2's byte-level BPE, how many symbols are in the base alphabet?",
            options: ["128", "256", "50,257", "All Unicode code points"],
            answer: 1,
            explain: "One per possible byte value: 256. The full vocabulary is 50,257.",
          },
          {
            q: "In BERT's WordPiece, the word `xylophone` contains a character not in the vocabulary. What comes out?",
            options: ["`x` `##ylophone`", "`[UNK]` for the unknown character only", "A single `[UNK]` for the whole word", "The word is dropped"],
            answer: 2,
            explain: "Greedy longest-match-first fails for the word, so BERT outputs one `[UNK]` for the entire word.",
          },
          {
            kind: "numeric",
            q: "An untied model has embedding dimension 512 and a vocabulary of 30,000. How many parameters (in millions) are in the input embedding matrix alone?",
            answer: 15.36,
            tolerance: 0.01,
            explain: "$30{,}000 \\times 512 = 15{,}360{,}000$ = 15.36 M. If not tied, the output softmax matrix adds the same again.",
          },
          {
            q: "What does the attention mask hold for a padded position?",
            options: ["1", "0", "The pad ID", "−100"],
            answer: 1,
            explain: "1 marks real tokens, 0 marks padding. (−100 is PyTorch's default `ignore_index` for targets — a different mechanism.)",
          },
          {
            q: "Which statement about SentencePiece is correct?",
            options: ["It requires whitespace pre-tokenization", "It replaces spaces with ▁ and treats them as ordinary characters", "It only implements WordPiece", "It cannot decode losslessly"],
            answer: 1,
            explain: "Treating spaces as the symbol ▁ makes SentencePiece language-independent and reversible.",
          },
        ],
      },
      {
        t: "exercise",
        title: "Numerical: run BPE by hand",
        prompt: "Corpus: `hug` ×10, `pug` ×5, `pun` ×12, `bun` ×4, `hugs` ×5. With an end-of-word marker `_` and alphabetical tie-breaking, do the first four BPE merges. Give each merge with its count, and the final split of every word.",
        hint: "Starting splits are `h u g _`, `p u g _`, `p u n _`, `b u n _`, `h u g s _`. Recount after every merge.",
        solution: [
          {
            t: "list",
            ordered: true,
            items: [
              "Counts: `u g` = 10 + 5 + 5 = 20, `p u` = 17, `n _` = 16, `u n` = 16, `g _` = 15, `h u` = 15. Merge **`u`+`g` → `ug` (20)**.",
              "`n _` = 16 and `u n` = 16 now tie; `n` < `u` alphabetically, so merge **`n`+`_` → `n_` (16)**.",
              "Now the top pair is `u n_` = 12 + 4 = 16. Merge **`u`+`n_` → `un_` (16)**.",
              "`h ug` = 15 and `ug _` = 15 tie; `h` < `ug`, so merge **`h`+`ug` → `hug` (15)**.",
            ],
          },
          {
            t: "p",
            text: "Final splits: `hug _`, `p ug _`, `p un_`, `b un_`, `hug s _`. (Computed and checked with the from-scratch code.)",
          },
        ],
      },
      {
        t: "exercise",
        title: "Coding: WordPiece greedy longest-match-first encoder",
        prompt: "Write BERT-style WordPiece encoding for one word: keep taking the longest vocabulary entry that matches at the current position (later pieces carry `##`); if nothing matches, return `[UNK]` for the whole word. Test it on a small vocabulary.",
        hint: "Two nested loops: the outer moves `start` forward; the inner shrinks `end` from the end of the word until `word[start:end]` (with `##` if `start > 0`) is in the vocabulary.",
        solution: [
          {
            t: "code",
            lang: "python",
            code: String.raw`def wordpiece_encode(word, vocab, unk="[UNK]", max_chars=100):
    """Greedy longest-match-first (as in BERT): repeatedly take the longest
    vocabulary entry that matches at the current position."""
    if len(word) > max_chars:
        return [unk]
    pieces, start = [], 0
    while start < len(word):
        end = len(word)
        match = None
        while end > start:
            piece = word[start:end]
            if start > 0:
                piece = "##" + piece          # continuation pieces carry the ## prefix
            if piece in vocab:
                match = piece
                break
            end -= 1
        if match is None:
            return [unk]                       # no piece matches: the whole word is unknown
        pieces.append(match)
        start = end
    return pieces

vocab = {"un", "##aff", "##able", "aff", "##ord", "##s", "play", "##ing", "##ed", "u", "##n", "##a", "##f"}
for w in ["unaffordable", "playing", "played", "unplayable", "plays", "xylophone"]:
    print(f"{w:13s} -> {wordpiece_encode(w, vocab)}")`,
            output: `unaffordable  -> ['un', '##aff', '##ord', '##able']
playing       -> ['play', '##ing']
played        -> ['play', '##ed']
unplayable    -> ['[UNK]']
plays         -> ['play', '##s']
xylophone     -> ['[UNK]']`,
          },
          {
            t: "p",
            text: "`unplayable` fails even though `play` is in the vocabulary: after `un`, the next piece would need to be `##play` (or shorter `##p…` pieces), which do not exist. Starting and continuation pieces are separate entries. Greedy matching also picks `un` over `u` + `##n` just because it is longer.",
          },
        ],
      },
    ],

    // ------------------------------------------------------------------ summary
    summary: [
      {
        t: "list",
        items: [
          "Pipeline: normalize → pre-tokenize → subword model → special tokens → pad/truncate + attention mask → IDs.",
          "Normalization: NFKC unifies look-alike characters and composed/decomposed forms; lowercasing and accent stripping trade information for a smaller vocabulary.",
          "Word-level: short sequences, but OOV and no sharing between related words. Character/byte-level: no OOV, but long sequences.",
          "BPE: merge the most frequent neighbouring pair, $\\arg\\max \\text{count}(a,b)$; encode by replaying merges in order. $V = |A| + M$.",
          "Byte-level BPE (GPT-2): 256-byte base alphabet, no `<unk>`, `Ġ` marks a leading space; 50,257 tokens.",
          "WordPiece (BERT): merge by $\\text{count}(ab)/(\\text{count}(a)\\text{count}(b))$; `##` continuation; greedy longest-match encoding; unknown ⇒ whole-word `[UNK]`.",
          "Unigram LM: top-down pruning by likelihood; Viterbi encoding; can sample splits (subword regularization). SentencePiece: raw text, spaces as `▁`, lossless.",
          "Vocabulary size trades $V \\times d$ embedding (and softmax) parameters against sequence length.",
          "Special tokens `[CLS]`, `[SEP]`, `[PAD]`, `<bos>`, `<eos>`; attention mask 1 = real, 0 = pad; truncate to the model's maximum length.",
        ],
      },
    ],

    // ------------------------------------------------------------------ next
    next: [
      {
        t: "list",
        items: [
          "[Word embeddings: Word2Vec, GloVe, FastText](/learn/m13/word-embeddings) — what to do with token IDs: learn a vector for each. FastText's character n-grams are another fix for OOV.",
          "[Tokenization, embeddings and context windows](/learn/m11/tokens-embeddings) — how token IDs enter a Transformer, and why context windows are measured in tokens.",
          "[Language modelling and perplexity](/learn/m13/language-modeling) — predicting the next token, and why perplexity depends on the tokenizer.",
        ],
      },
    ],
  },
};

export default lesson;
