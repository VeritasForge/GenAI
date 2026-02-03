"""TDD tests for Pipeline orchestration (LangChain)."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_langchain.pipeline import (
    AskResult,
    IndexResult,
    ask_question,
    index_documents,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def _make_data_dir(tmp_path: Path, file_count: int = 2) -> Path:
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


# --- Group A: Basic behavior ---


class TestBasicBehavior:
    def test_should_return_index_result(self, tmp_path):
        """Cycle 1: index_documents returns IndexResult with correct fields."""
        # Given
        data_dir = _make_data_dir(tmp_path, file_count=2)
        db_path = str(tmp_path / "chroma_db")

        # When
        result = index_documents(
            data_dir=data_dir,
            db_path=db_path,
            collection_name="test_index",
        )

        # Then
        assert isinstance(result, IndexResult)
        assert result.total_documents == 2
        assert result.total_chunks > 0
        assert result.db_path == db_path
        assert result.collection_name == "test_index"

    @patch("rag_langchain.llm.subprocess.run")
    def test_should_return_ask_result(self, mock_run, tmp_path):
        """Cycle 2: ask_question returns AskResult with mocked subprocess."""
        # Given
        mock_run.return_value = _mock_subprocess_success("Test answer")
        data_dir = _make_data_dir(tmp_path)
        db_path = str(tmp_path / "chroma_db")
        index_documents(data_dir=data_dir, db_path=db_path, collection_name="test_ask")

        # When
        result = ask_question(
            query="What is document 0?",
            db_path=db_path,
            collection_name="test_ask",
        )

        # Then
        assert isinstance(result, AskResult)
        assert result.answer == "Test answer"

    @patch("rag_langchain.llm.subprocess.run")
    def test_should_persist_and_query(self, mock_run, tmp_path):
        """Cycle 3: index and ask share the same persist directory."""
        # Given
        mock_run.return_value = _mock_subprocess_success("Persistent answer")
        data_dir = _make_data_dir(tmp_path)
        db_path = str(tmp_path / "chroma_db")
        collection = "test_persist"

        # When
        index_documents(data_dir=data_dir, db_path=db_path, collection_name=collection)
        result = ask_question(
            query="document",
            db_path=db_path,
            collection_name=collection,
        )

        # Then
        assert result.answer == "Persistent answer"
        assert len(result.sources) > 0


# --- Group B: Parameters ---


class TestParameters:
    def test_should_chunk_size_affect_count(self, tmp_path):
        """Cycle 4: smaller chunk_size produces more chunks."""
        # Given
        data_dir = _make_data_dir(tmp_path, file_count=1)

        # When
        r_large = index_documents(
            data_dir=data_dir,
            db_path=str(tmp_path / "db_large"),
            collection_name="large",
            chunk_size=500,
        )
        r_small = index_documents(
            data_dir=data_dir,
            db_path=str(tmp_path / "db_small"),
            collection_name="small",
            chunk_size=200,
        )

        # Then
        assert r_small.total_chunks > r_large.total_chunks

    @patch("rag_langchain.llm.subprocess.run")
    def test_should_top_k_limit_sources(self, mock_run, tmp_path):
        """Cycle 5: top_k=2 limits sources to at most 2."""
        # Given
        mock_run.return_value = _mock_subprocess_success("answer")
        data_dir = _make_data_dir(tmp_path, file_count=3)
        db_path = str(tmp_path / "chroma_db")
        index_documents(data_dir=data_dir, db_path=db_path, collection_name="topk")

        # When
        result = ask_question(
            query="document",
            db_path=db_path,
            collection_name="topk",
            top_k=2,
        )

        # Then
        assert len(result.sources) <= 2


# --- Group C: Edge cases ---


class TestEdgeCases:
    def test_should_raise_on_nonexistent_dir(self):
        """Cycle 6: non-existent directory raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            index_documents(data_dir="/nonexistent/path")

    def test_should_raise_on_empty_dir(self, tmp_path):
        """Cycle 7: directory with no .txt files raises ValueError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No .txt files"):
            index_documents(data_dir=empty_dir)

    def test_should_raise_on_empty_query(self):
        """Cycle 8: empty string raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            ask_question(query="")

    def test_should_raise_on_whitespace_query(self):
        """Cycle 9: whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            ask_question(query="   ")


# --- Group D: Integration ---


class TestIntegration:
    @pytest.mark.skipif(
        shutil.which("claude") is None,
        reason="claude CLI not found in PATH",
    )
    def test_should_full_index_and_ask(self, tmp_path):
        """Cycle 10: full E2E with real data and claude CLI."""
        db_path = str(tmp_path / "chroma_db")

        result = index_documents(
            data_dir=DATA_DIR,
            db_path=db_path,
            collection_name="e2e",
            chunk_size=500,
            chunk_overlap=50,
        )
        assert result.total_documents >= 1
        assert result.total_chunks >= 1

        ask_result = ask_question(
            query="What are the side effects of metformin?",
            db_path=db_path,
            collection_name="e2e",
        )
        assert len(ask_result.answer) > 0
        assert len(ask_result.sources) > 0
