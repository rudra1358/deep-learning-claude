"""All settings in one place, overridable with environment variables prefixed DI_
(e.g. DI_DEVICE=cpu, DI_ENABLE_GENERATOR=false).

The ML service has no database: documents, pages and chats live in MongoDB, owned by the Node
server. The only state kept here is the search index (FAISS vectors + the chunk store), in
data/index/ - the same job a vector database does."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]          # projects/doc-intelligence


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DI_", extra="ignore")

    data_dir: Path = ROOT / "data"
    models_dir: Path = ROOT / "models"                       # our fine-tuned NER + classifier
    metrics_file: Path = ROOT / "training" / "reports" / "metrics.json"
    device: str = "auto"

    embedder: str = "BAAI/bge-small-en-v1.5"
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    qa_model: str = "deepset/roberta-base-squad2"
    summarizer: str = "sshleifer/distilbart-cnn-12-6"
    generator: str = "Qwen/Qwen2.5-1.5B-Instruct"
    enable_generator: bool = True                            # set false on machines without a GPU
    generator_gate: bool = True      # ask the extractive reader first; skip the LLM if it finds no answer

    chunk_tokens: int = 256          # chosen by the chunk-size experiment (training/eval_retrieval.py)
    chunk_overlap: int = 40
    retrieve_k: int = 30             # candidates from hybrid search ...
    rerank_k: int = 5                # ... of which the cross-encoder keeps this many
    no_answer_margin: float | None = None   # None -> use the value tuned in training/eval_qa.py


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    (s.data_dir / "index").mkdir(parents=True, exist_ok=True)
    return s
