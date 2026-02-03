"""TDD tests for Query module (LlamaIndex)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_llamaindex.ingestion import create_index, load_and_split
from rag_llamaindex.query import QueryResult, query_index


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


def _create_test_index(tmp_path: Path):
    """Helper to create a test index."""
    data_dir = _make_data_dir(tmp_path)
    nodes = load_and_split(data_dir, chunk_size=500)
    db_path = str(tmp_path / "chroma_db")
    return create_index(nodes=nodes, db_path=db_path, collection_name="test_query")


# --- Group A: Basic Query ---


class TestBasicQuery:
    @patch("rag_llamaindex.llm.subprocess.run")
    def test_should_return_query_result(self, mock_run, tmp_path):
        """Cycle 1: query_index returns QueryResult with answer and sources."""
        # Given
        mock_run.return_value = _mock_subprocess_success("This is the answer.")
        index = _create_test_index(tmp_path)

        # When
        result = query_index(index, "What is document 0?")

        # Then
        assert isinstance(result, QueryResult)
        assert result.answer == "This is the answer."
        assert len(result.sources) > 0

    @patch("rag_llamaindex.llm.subprocess.run")
    def test_should_include_source_metadata(self, mock_run, tmp_path):
        """Cycle 2: sources contain file_name metadata."""
        # Given
        mock_run.return_value = _mock_subprocess_success("Answer")
        index = _create_test_index(tmp_path)

        # When
        result = query_index(index, "document")

        # Then
        assert len(result.sources) > 0
        first_source = result.sources[0]
        assert "file_name" in first_source.metadata

    @patch("rag_llamaindex.llm.subprocess.run")
    def test_should_include_source_text(self, mock_run, tmp_path):
        """Cycle 3: sources contain text content."""
        # Given
        mock_run.return_value = _mock_subprocess_success("Answer")
        index = _create_test_index(tmp_path)

        # When
        result = query_index(index, "document")

        # Then
        assert len(result.sources) > 0
        assert len(result.sources[0].text) > 0

    @patch("rag_llamaindex.llm.subprocess.run")
    def test_should_include_source_score(self, mock_run, tmp_path):
        """Cycle 4: sources contain similarity scores."""
        # Given
        mock_run.return_value = _mock_subprocess_success("Answer")
        index = _create_test_index(tmp_path)

        # When
        result = query_index(index, "document")

        # Then
        assert len(result.sources) > 0
        assert result.sources[0].score is not None


# --- Group B: Parameters ---


class TestParameters:
    @patch("rag_llamaindex.llm.subprocess.run")
    def test_should_respect_top_k(self, mock_run, tmp_path):
        """Cycle 5: top_k limits number of sources."""
        # Given
        mock_run.return_value = _mock_subprocess_success("Answer")
        data_dir = _make_data_dir(tmp_path, file_count=3)
        nodes = load_and_split(data_dir, chunk_size=200)
        db_path = str(tmp_path / "chroma_db")
        index = create_index(nodes=nodes, db_path=db_path, collection_name="topk")

        # When
        result = query_index(index, "document", top_k=2)

        # Then
        assert len(result.sources) <= 2


# --- Group C: Edge Cases ---


class TestEdgeCases:
    def test_should_raise_on_empty_query(self, tmp_path):
        """Cycle 6: empty query raises ValueError."""
        # Given
        index = _create_test_index(tmp_path)

        # When / Then
        with pytest.raises(ValueError, match="Query cannot be empty"):
            query_index(index, "")

    def test_should_raise_on_whitespace_query(self, tmp_path):
        """Cycle 7: whitespace-only query raises ValueError."""
        # Given
        index = _create_test_index(tmp_path)

        # When / Then
        with pytest.raises(ValueError, match="Query cannot be empty"):
            query_index(index, "   ")
