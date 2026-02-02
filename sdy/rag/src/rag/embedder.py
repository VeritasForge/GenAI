"""Embedding — Step 3 of the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from rag.loader import Document

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformers model (singleton)."""
    global _model  # noqa: PLW0603
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


@dataclass
class EmbeddedDocument:
    """A document chunk paired with its embedding vector."""

    document: Document
    embedding: list[float]


def embed_query(text: str) -> list[float]:
    """Embed a single query string into a 384-dim vector."""
    model = _get_model()
    vector = model.encode(text)
    return vector.tolist()


def embed_documents(docs: list[Document]) -> list[EmbeddedDocument]:
    """Embed multiple Document chunks in a single batch."""
    if not docs:
        return []

    for doc in docs:
        if not doc.content.strip():
            raise ValueError("Document content is empty")

    model = _get_model()
    texts = [doc.content for doc in docs]
    vectors = model.encode(texts)

    return [
        EmbeddedDocument(document=doc, embedding=vec.tolist())
        for doc, vec in zip(docs, vectors)
    ]
