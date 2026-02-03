"""TDD tests for Ingestion module (LlamaIndex)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_llamaindex.ingestion import create_index, load_and_split


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
        """Cycle 1: load_and_split returns nodes from .txt files."""
        # Given
        data_dir = _make_data_dir(tmp_path, file_count=2)

        # When
        nodes = load_and_split(data_dir)

        # Then
        assert len(nodes) > 0

    def test_should_split_into_chunks(self, tmp_path):
        """Cycle 2: smaller chunk_size produces more nodes."""
        # Given
        data_dir = _make_data_dir(tmp_path, file_count=1)

        # When
        nodes_large = load_and_split(data_dir, chunk_size=500)
        nodes_small = load_and_split(data_dir, chunk_size=200)

        # Then
        assert len(nodes_small) > len(nodes_large)

    def test_should_raise_on_nonexistent_directory(self):
        """Cycle 3: non-existent directory raises ValueError."""
        # When / Then
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
        """Cycle 5: nodes contain file path metadata."""
        # Given
        data_dir = _make_data_dir(tmp_path, file_count=1)

        # When
        nodes = load_and_split(data_dir)

        # Then
        assert len(nodes) > 0
        first_node = nodes[0]
        assert first_node.metadata is not None
        assert "file_name" in first_node.metadata


# --- Group B: Create Index ---


class TestCreateIndex:
    def test_should_create_index_from_nodes(self, tmp_path):
        """Cycle 6: create_index stores nodes and returns VectorStoreIndex."""
        # Given
        from llama_index.core import VectorStoreIndex

        data_dir = _make_data_dir(tmp_path, file_count=1)
        nodes = load_and_split(data_dir, chunk_size=500)
        db_path = str(tmp_path / "chroma_db")

        # When
        index = create_index(
            nodes=nodes,
            db_path=db_path,
            collection_name="test_create",
        )

        # Then
        assert isinstance(index, VectorStoreIndex)

    def test_should_persist_to_disk(self, tmp_path):
        """Cycle 7: create_index creates chroma_db directory."""
        # Given
        data_dir = _make_data_dir(tmp_path, file_count=1)
        nodes = load_and_split(data_dir, chunk_size=500)
        db_path = str(tmp_path / "chroma_db")

        # When
        create_index(nodes=nodes, db_path=db_path, collection_name="test_persist")

        # Then
        assert Path(db_path).exists()

    def test_should_load_existing_index(self, tmp_path):
        """Cycle 8: create_index with existing db loads persisted data."""
        # Given
        data_dir = _make_data_dir(tmp_path, file_count=1)
        nodes = load_and_split(data_dir, chunk_size=500)
        db_path = str(tmp_path / "chroma_db")

        # First: create and persist
        create_index(nodes=nodes, db_path=db_path, collection_name="test_load")

        # When: load from existing without adding nodes
        index = create_index(
            nodes=None,
            db_path=db_path,
            collection_name="test_load",
        )

        # Then
        retriever = index.as_retriever(similarity_top_k=1)
        results = retriever.retrieve("document")
        assert len(results) > 0
