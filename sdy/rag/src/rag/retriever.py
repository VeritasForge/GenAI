"""Retrieval — Step 5 of the RAG pipeline."""

from __future__ import annotations

from rag.embedder import embed_query
from rag.store import SearchResult, VectorStore


class Retriever:
    """High-level retrieval facade: query string → relevant documents."""

    def __init__(
        self,
        store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> None:
        self._store = store
        self._top_k = top_k
        self._score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant documents for a natural-language query."""
        query_embedding = embed_query(query)
        k = top_k if top_k is not None else self._top_k
        threshold = (
            score_threshold if score_threshold is not None else self._score_threshold
        )

        return self._store.search(
            query_embedding, top_k=k, score_threshold=threshold
        )
