"""Turns text into vectors so that "similar meaning" becomes "nearby vectors".

A bi-encoder (Sentence-Transformers) encodes the question and each chunk SEPARATELY. That is
what makes search fast: chunk vectors are computed once at upload time and stored; at query
time we only encode the short question and compare it with every stored vector.

Vectors are L2-normalised, so the dot product of two vectors equals their cosine similarity
(1 = same direction / meaning, 0 = unrelated).

Some models (BGE) were trained with an instruction prefix on the QUERY side only; using the
same prefix at search time matters for their accuracy, so it is part of the config here.
"""
from __future__ import annotations

import numpy as np

QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        # renamed in sentence-transformers 6 (old name still works but warns)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.max_tokens = self.model.max_seq_length
        self.tokenizer = self.model.tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        q = QUERY_PREFIX.get(self.name, "") + text
        return self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).astype(np.float32)[0]
