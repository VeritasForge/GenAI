"""TDD tests for Pipeline orchestration (Step 7)."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.pipeline import AskResult, IndexResult, ask_question, index_documents

DATA_DIR = Path(__file__).parent.parent / "data"


def _make_data_dir(tmp_path: Path, file_count: int = 2) -> Path:
    """Create a temp directory with sample .txt files."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i in range(file_count):
        (tmp_path / f"doc_{i}.txt").write_text(
            f"This is document number {i}. " * 20, encoding="utf-8"
        )
    return tmp_path


def _mock_subprocess_success(answer: str = "Mock answer") -> MagicMock:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = answer
    mock_result.stderr = ""
    return mock_result


# --- Group A: Basic behavior ---


class TestBasicBehavior:
    def test_index_returns_result(self, tmp_path):
        """Cycle 1: index_documents returns IndexResult with correct fields."""
        data_dir = _make_data_dir(tmp_path / "data", file_count=2)
        db_path = str(tmp_path / "chroma_db")

        result = index_documents(
            data_dir=data_dir,
            db_path=db_path,
            collection_name="test_index",
        )

        assert isinstance(result, IndexResult)
        assert result.total_documents == 2
        assert result.total_chunks > 0
        assert result.db_path == db_path
        assert result.collection_name == "test_index"

    @patch("rag.generator.subprocess.run")
    def test_ask_returns_result(self, mock_run, tmp_path):
        """Cycle 2: ask_question returns AskResult with mocked subprocess."""
        mock_run.return_value = _mock_subprocess_success("Test answer")

        # Index first
        data_dir = _make_data_dir(tmp_path / "data")
        db_path = str(tmp_path / "chroma_db")
        index_documents(data_dir=data_dir, db_path=db_path, collection_name="test_ask")

        result = ask_question(
            query="What is document 0?",
            db_path=db_path,
            collection_name="test_ask",
        )

        assert isinstance(result, AskResult)
        assert result.answer == "Test answer"
        assert result.model == "claude-code"

    @patch("rag.generator.subprocess.run")
    def test_index_then_ask_persistence(self, mock_run, tmp_path):
        """Cycle 3: index and ask share the same persist directory."""
        mock_run.return_value = _mock_subprocess_success("Persistent answer")

        data_dir = _make_data_dir(tmp_path / "data")
        db_path = str(tmp_path / "chroma_db")
        collection = "test_persist"

        index_documents(data_dir=data_dir, db_path=db_path, collection_name=collection)
        result = ask_question(
            query="document",
            db_path=db_path,
            collection_name=collection,
        )

        assert result.answer == "Persistent answer"
        assert len(result.sources) > 0


# --- Group B: Parameters ---


class TestParameters:
    def test_chunk_size_affects_count(self, tmp_path):
        """Cycle 4: smaller chunk_size produces more chunks."""
        data_dir = _make_data_dir(tmp_path / "data", file_count=1)

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

        assert r_small.total_chunks > r_large.total_chunks

    @patch("rag.generator.subprocess.run")
    def test_top_k_limits_sources(self, mock_run, tmp_path):
        """Cycle 5: top_k=2 limits sources to at most 2."""
        mock_run.return_value = _mock_subprocess_success("answer")

        data_dir = _make_data_dir(tmp_path / "data", file_count=3)
        db_path = str(tmp_path / "chroma_db")
        index_documents(data_dir=data_dir, db_path=db_path, collection_name="topk")

        result = ask_question(
            query="document",
            db_path=db_path,
            collection_name="topk",
            top_k=2,
        )

        assert len(result.sources) <= 2

    @patch("rag.generator.subprocess.run")
    def test_score_threshold_filters(self, mock_run, tmp_path):
        """Cycle 6: high threshold filters out low-score sources."""
        mock_run.return_value = _mock_subprocess_success("answer")

        data_dir = _make_data_dir(tmp_path / "data")
        db_path = str(tmp_path / "chroma_db")
        index_documents(data_dir=data_dir, db_path=db_path, collection_name="thresh")

        result_no_filter = ask_question(
            query="document number 0",
            db_path=db_path,
            collection_name="thresh",
            score_threshold=0.0,
        )
        result_high_filter = ask_question(
            query="document number 0",
            db_path=db_path,
            collection_name="thresh",
            score_threshold=0.99,
        )

        assert len(result_high_filter.sources) <= len(result_no_filter.sources)

    @patch("rag.generator.subprocess.run")
    def test_custom_system_prompt(self, mock_run, tmp_path):
        """Cycle 7: custom system_prompt is passed through to CLI call."""
        mock_run.return_value = _mock_subprocess_success("custom answer")

        data_dir = _make_data_dir(tmp_path / "data")
        db_path = str(tmp_path / "chroma_db")
        index_documents(data_dir=data_dir, db_path=db_path, collection_name="prompt")

        ask_question(
            query="test",
            db_path=db_path,
            collection_name="prompt",
            system_prompt="You are a medical expert.",
        )

        call_args = mock_run.call_args
        prompt_arg = call_args[0][0][2]  # claude -p <prompt>
        assert "You are a medical expert." in prompt_arg


# --- Group C: Edge cases ---


class TestEdgeCases:
    def test_index_nonexistent_dir(self):
        """Cycle 8: non-existent directory raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            index_documents(data_dir="/nonexistent/path")

    def test_index_empty_dir(self, tmp_path):
        """Cycle 9: directory with no .txt files raises ValueError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No .txt files found"):
            index_documents(data_dir=empty_dir)

    def test_ask_empty_query(self):
        """Cycle 10: empty string raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            ask_question(query="")

    def test_ask_whitespace_query(self):
        """Cycle 11: whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            ask_question(query="   ")


# --- Group D: Integration ---


class TestIntegration:
    @pytest.mark.skipif(
        shutil.which("claude") is None,
        reason="claude CLI not found in PATH",
    )
    def test_full_index_and_ask(self, tmp_path):
        """Cycle 12: full E2E with real data and claude CLI."""
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
