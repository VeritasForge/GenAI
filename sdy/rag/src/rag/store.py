"""Vector Store — Step 4 of the RAG pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import chromadb

from rag.embedder import EmbeddedDocument
from rag.loader import Document


@dataclass
class SearchResult:
    """A search result: document + similarity score."""

    document: Document
    score: float


class VectorStore:
    """ChromaDB-backed vector store for document embeddings."""

    def __init__(
        self,
        collection_name: str = "rag",
        persist_directory: str | None = None,
    ) -> None:
        if persist_directory is None:
            self._client = chromadb.Client()
        else:
            self._client = chromadb.PersistentClient(path=persist_directory)

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, docs: list[EmbeddedDocument]) -> list[str]:
        """Upsert embedded documents into the collection. Returns IDs."""
        if not docs:
            return []

        ids = [self._make_id(doc) for doc in docs]
        embeddings = [doc.embedding for doc in docs]
        documents = [doc.document.content for doc in docs]
        metadatas = [{k: v for k, v in doc.document.metadata.items()} for doc in docs]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return ids

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Search for similar documents by query vector.

        Args:
            query_embedding: The query vector.
            top_k: Maximum number of results to return.
            score_threshold: Minimum similarity score (0.0 = no filtering).
        """
        if self._collection.count() == 0:
            return []

        actual_k = min(top_k, self._collection.count())
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        for doc_text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - distance
            if score_threshold > 0.0 and score < score_threshold:
                continue
            document = Document(content=doc_text, metadata=dict(metadata))
            search_results.append(SearchResult(document=document, score=score))

        return search_results

    def count(self) -> int:
        """Return the number of stored documents."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete all documents in the collection (for testing)."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _make_id(doc: EmbeddedDocument) -> str:
        """Generate a deterministic ID from metadata, or UUID fallback."""
        meta = doc.document.metadata
        filename = meta.get("filename")
        chunk_index = meta.get("chunk_index")
        if filename is not None and chunk_index is not None:
            return f"{filename}::chunk-{chunk_index}"
        return str(uuid.uuid4())
