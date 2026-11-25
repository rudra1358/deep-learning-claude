"""Step 0: download every model and dataset once, so later steps work offline.

Models (all open-weight, run locally):
  embedders      sentence-transformers/all-MiniLM-L6-v2 (22 M params), BAAI/bge-small-en-v1.5 (33 M)
  re-ranker      cross-encoder/ms-marco-MiniLM-L-6-v2 (22 M)
  extractive QA  deepset/roberta-base-squad2 (125 M) - trained to say "no answer" too
  summariser     sshleifer/distilbart-cnn-12-6 (306 M)
  generator      Qwen/Qwen2.5-1.5B-Instruct (1.5 B) - optional, written answers with citations
  base encoder   distilbert-base-cased (66 M) - we fine-tune it ourselves for NER + classification
Datasets: SQuAD 2.0 (dev), CoNLL-2003 (NER), AG News (topic classification), CNN/DailyMail (summaries).

    python training/download_assets.py
"""
from datasets import load_dataset
from huggingface_hub import snapshot_download

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "deepset/roberta-base-squad2",
    "sshleifer/distilbart-cnn-12-6",
    "distilbert-base-cased",
    "Qwen/Qwen2.5-1.5B-Instruct",
]
DATASETS = [
    ("rajpurkar/squad_v2", None, "validation"),
    ("eriktks/conll2003", None, None),
    ("fancyzhx/ag_news", None, None),
    ("abisee/cnn_dailymail", "3.0.0", "test"),
]

if __name__ == "__main__":
    for m in MODELS:
        patterns = ["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "vocab*", "*tokenizer*",
                    "1_Pooling/*", "modules.json"]
        if m == "sshleifer/distilbart-cnn-12-6":       # this repo has no .safetensors, only the PyTorch .bin
            patterns.append("pytorch_model.bin")
        path = snapshot_download(m, allow_patterns=patterns)
        print("model  ", m, "->", path)
    for name, cfg, split in DATASETS:
        rev = "refs/convert/parquet" if name == "eriktks/conll2003" else None   # see finetune_ner.py
        ds = load_dataset(name, cfg, split=split, revision=rev)
        print("dataset", name, cfg or "", split or "", "->", ds)
