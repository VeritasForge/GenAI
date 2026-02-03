"""TDD tests for Ingestion module (LangChain)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_langchain.ingestion import create_vector_store, load_and_split


def _make_data_dir(tmp_path: Path, file_count: int = 2) -> Path:
    """Create a temp directory with sample .txt files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(file_count):
        (data_dir / f"doc_{i}.txt").write_text(
            f"This is document number {i}. " * 50, encoding="utf-8"
        )
    return data_dir


# --- Group A: Load and Split ---


class TestLoadAndSplit:
    def test_should_load_documents_from_directory(self, tmp_path):
        """Cycle 1: load_and_split returns Document list from .txt files."""
        # Given
        data_dir = _make_data_dir(tmp_path, file_count=2)

        # When
        docs = load_and_split(data_dir)

        # Then
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_should_split_into_chunks(self, tmp_path):
        """Cycle 2: smaller chunk_size produces more documents."""
        # Given
        data_dir = _make_data_dir(tmp_path, file_count=1)

        # When
        docs_large = load_and_split(data_dir, chunk_size=500)
        docs_small = load_and_split(data_dir, chunk_size=200)

        # Then
        assert len(docs_small) > len(docs_large)

    def test_should_raise_on_nonexistent_directory(self):
        """Cycle 3: non-existent directory raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            load_and_split(Path("/nonexistent/path"))

    def test_should_raise_on_empty_directory(self, tmp_path):
        """Cycle 4: directory with no .txt files raises ValueError."""
        # Given
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        # When / Then
        with pytest.raises(ValueError, match="No .txt files"):
            load_and_split(empty_dir)

    def test_should_preserve_metadata(self, tmp_path):
        """Cycle 5: documents contain source metadata."""
        # Given
        data_dir = _make_data_dir(tmp_path, file_count=1)

        # When
        docs = load_and_split(data_dir)

        # Then
        assert len(docs) > 0
        assert "source" in docs[0].metadata


# --- Group B: Create Vector Store ---


class TestCreateVectorStore:
    def test_should_create_store_from_documents(self, tmp_path):
        """Cycle 6: create_vector_store stores documents and returns Chroma."""
        # Given
        from langchain_chroma import Chroma

        data_dir = _make_data_dir(tmp_path, file_count=1)
        docs = load_and_split(data_dir, chunk_size=500)
        db_path = str(tmp_path / "chroma_db")

        # When
        store = create_vector_store(
            documents=docs,
            db_path=db_path,
            collection_name="test_create",
        )

        # Then
        assert isinstance(store, Chroma)

    def test_should_persist_to_disk(self, tmp_path):
        """Cycle 7: create_vector_store creates chroma_db directory."""
        # Given
        data_dir = _make_data_dir(tmp_path, file_count=1)
        docs = load_and_split(data_dir, chunk_size=500)
        db_path = str(tmp_path / "chroma_db")

        # When
        create_vector_store(
            documents=docs, db_path=db_path, collection_name="test_persist"
        )

        # Then
        assert Path(db_path).exists()

    def test_should_load_existing_store(self, tmp_path):
        """Cycle 8: create_vector_store with no documents loads persisted data."""
        # Given
        data_dir = _make_data_dir(tmp_path, file_count=1)
        docs = load_and_split(data_dir, chunk_size=500)
        db_path = str(tmp_path / "chroma_db")

        # First: create and persist
        create_vector_store(
            documents=docs, db_path=db_path, collection_name="test_load"
        )

        # When: load from existing
        store = create_vector_store(
            documents=None, db_path=db_path, collection_name="test_load"
        )

        # Then
        results = store.similarity_search("document", k=1)
        assert len(results) > 0
