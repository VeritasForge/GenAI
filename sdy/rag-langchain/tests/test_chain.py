"""TDD tests for LCEL RAG Chain (LangChain)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_langchain.chain import QueryResult, build_rag_chain, query_with_chain
from rag_langchain.ingestion import create_vector_store, load_and_split


def _make_data_dir(tmp_path: Path, file_count: int = 1) -> Path:
    """Create a temp directory with sample .txt files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(file_count):
        (data_dir / f"doc_{i}.txt").write_text(
            f"This is document number {i}. " * 50, encoding="utf-8"
        )
    return data_dir


def _mock_subprocess_success(answer: str = "Mock answer") -> MagicMock:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = answer
    mock_result.stderr = ""
    return mock_result


def _create_test_store(tmp_path: Path):
    """Helper to create a test vector store."""
    data_dir = _make_data_dir(tmp_path)
    docs = load_and_split(data_dir, chunk_size=500)
    db_path = str(tmp_path / "chroma_db")
    return create_vector_store(
        documents=docs, db_path=db_path, collection_name="test_chain"
    )


# --- Group A: Build Chain ---


class TestBuildChain:
    def test_should_build_rag_chain(self, tmp_path):
        """Cycle 1: build_rag_chain returns a Runnable chain."""
        # Given
        store = _create_test_store(tmp_path)

        # When
        chain = build_rag_chain(store, top_k=3)

        # Then
        assert chain is not None
        assert hasattr(chain, "invoke")


# --- Group B: Query ---


class TestQuery:
    @patch("rag_langchain.llm.subprocess.run")
    def test_should_return_query_result(self, mock_run, tmp_path):
        """Cycle 2: query_with_chain returns QueryResult with answer and sources."""
        # Given
        mock_run.return_value = _mock_subprocess_success("This is the answer.")
        store = _create_test_store(tmp_path)

        # When
        result = query_with_chain(store, "What is document 0?")

        # Then
        assert isinstance(result, QueryResult)
        assert result.answer == "This is the answer."
        assert len(result.sources) > 0

    @patch("rag_langchain.llm.subprocess.run")
    def test_should_include_source_metadata(self, mock_run, tmp_path):
        """Cycle 3: sources contain source file metadata."""
        # Given
        mock_run.return_value = _mock_subprocess_success("Answer")
        store = _create_test_store(tmp_path)

        # When
        result = query_with_chain(store, "document")

        # Then
        assert len(result.sources) > 0
        assert "source" in result.sources[0].metadata

    @patch("rag_langchain.llm.subprocess.run")
    def test_should_include_source_text(self, mock_run, tmp_path):
        """Cycle 4: sources contain text content."""
        # Given
        mock_run.return_value = _mock_subprocess_success("Answer")
        store = _create_test_store(tmp_path)

        # When
        result = query_with_chain(store, "document")

        # Then
        assert len(result.sources) > 0
        assert len(result.sources[0].text) > 0


# --- Group C: Parameters ---


class TestParameters:
    @patch("rag_langchain.llm.subprocess.run")
    def test_should_respect_top_k(self, mock_run, tmp_path):
        """Cycle 5: top_k limits number of sources."""
        # Given
        mock_run.return_value = _mock_subprocess_success("Answer")
        data_dir = _make_data_dir(tmp_path, file_count=3)
        docs = load_and_split(data_dir, chunk_size=200)
        db_path = str(tmp_path / "chroma_db")
        store = create_vector_store(
            documents=docs, db_path=db_path, collection_name="topk"
        )

        # When
        result = query_with_chain(store, "document", top_k=2)

        # Then
        assert len(result.sources) <= 2


# --- Group D: Edge Cases ---


class TestEdgeCases:
    def test_should_raise_on_empty_query(self, tmp_path):
        """Cycle 6: empty query raises ValueError."""
        # Given
        store = _create_test_store(tmp_path)

        # When / Then
        with pytest.raises(ValueError, match="Query cannot be empty"):
            query_with_chain(store, "")

    def test_should_raise_on_whitespace_query(self, tmp_path):
        """Cycle 7: whitespace-only query raises ValueError."""
        # Given
        store = _create_test_store(tmp_path)

        # When / Then
        with pytest.raises(ValueError, match="Query cannot be empty"):
            query_with_chain(store, "   ")
