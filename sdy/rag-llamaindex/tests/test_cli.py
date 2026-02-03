"""TDD tests for CLI commands (LlamaIndex)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from rag_llamaindex.cli import app
from rag_llamaindex.pipeline import AskResult, IndexResult
from rag_llamaindex.query import SourceInfo

runner = CliRunner()

DATA_DIR = Path(__file__).parent.parent / "data"


def _make_index_result(**kwargs) -> IndexResult:
    defaults = {
        "total_documents": 3,
        "total_chunks": 69,
        "db_path": "./chroma_db",
        "collection_name": "rag",
    }
    defaults.update(kwargs)
    return IndexResult(**defaults)


def _make_ask_result(**kwargs) -> AskResult:
    defaults = {
        "answer": "Metformin is a diabetes medication.",
        "sources": [
            SourceInfo(
                text="metformin treats diabetes",
                metadata={"file_name": "metformin.txt"},
                score=0.55,
            ),
            SourceInfo(
                text="side effects include nausea",
                metadata={"file_name": "metformin.txt"},
                score=0.48,
            ),
        ],
    }
    defaults.update(kwargs)
    return AskResult(**defaults)


# --- Group A: Basic commands ---


class TestBasicCommands:
    def test_should_hello(self):
        """Cycle 1: hello command works."""
        result = runner.invoke(app, ["hello"])
        assert result.exit_code == 0
        assert "RAG pipeline is ready" in result.output

    @patch("rag_llamaindex.cli.index_documents")
    def test_should_index_call_pipeline(self, mock_index):
        """Cycle 2: index command calls index_documents with correct args."""
        mock_index.return_value = _make_index_result()

        result = runner.invoke(
            app,
            ["index", "--data-dir", "./mydata", "--chunk-size", "300"],
        )

        assert result.exit_code == 0
        mock_index.assert_called_once()
        call_kwargs = mock_index.call_args[1]
        assert call_kwargs["data_dir"] == "./mydata"
        assert call_kwargs["chunk_size"] == 300

    @patch("rag_llamaindex.cli.ask_question")
    def test_should_ask_call_pipeline(self, mock_ask):
        """Cycle 3: ask command calls ask_question with query."""
        mock_ask.return_value = _make_ask_result()

        result = runner.invoke(app, ["ask", "What is metformin?"])

        assert result.exit_code == 0
        mock_ask.assert_called_once()
        call_kwargs = mock_ask.call_args[1]
        assert call_kwargs["query"] == "What is metformin?"


# --- Group B: Output format ---


class TestOutputFormat:
    @patch("rag_llamaindex.cli.index_documents")
    def test_should_show_index_counts(self, mock_index):
        """Cycle 4: index output includes document and chunk counts."""
        mock_index.return_value = _make_index_result(total_documents=3, total_chunks=69)

        result = runner.invoke(app, ["index"])

        assert "Loaded 3 documents" in result.output
        assert "69 chunks" in result.output
        assert "Done!" in result.output

    @patch("rag_llamaindex.cli.ask_question")
    def test_should_show_sources(self, mock_ask):
        """Cycle 5: ask output includes numbered source references."""
        mock_ask.return_value = _make_ask_result()

        result = runner.invoke(app, ["ask", "test query"])

        assert "[1]" in result.output
        assert "[2]" in result.output
        assert "metformin.txt" in result.output

    @patch("rag_llamaindex.cli.ask_question")
    def test_should_show_answer(self, mock_ask):
        """Cycle 6: ask output includes the answer text."""
        mock_ask.return_value = _make_ask_result(answer="This is the answer.")

        result = runner.invoke(app, ["ask", "test query"])

        assert "This is the answer." in result.output
        assert "Answer:" in result.output


# --- Group C: Error handling ---


class TestErrorHandling:
    @patch("rag_llamaindex.cli.index_documents")
    def test_should_handle_index_error(self, mock_index):
        """Cycle 7: ValueError from pipeline shows error and exits 1."""
        mock_index.side_effect = ValueError("Directory does not exist: /bad")

        result = runner.invoke(app, ["index", "--data-dir", "/bad"])

        assert result.exit_code == 1
        assert "Error:" in result.output

    def test_should_require_ask_query(self):
        """Cycle 8: missing query argument exits with non-zero."""
        result = runner.invoke(app, ["ask"])
        assert result.exit_code != 0

    @patch("rag_llamaindex.cli.ask_question")
    def test_should_handle_runtime_error(self, mock_ask):
        """Cycle 9: RuntimeError from pipeline shows error and exits 1."""
        mock_ask.side_effect = RuntimeError("Claude Code CLI failed")

        result = runner.invoke(app, ["ask", "some query"])

        assert result.exit_code == 1
        assert "Error:" in result.output


# --- Group D: Integration ---


class TestIntegration:
    def test_should_full_index_via_cli(self, tmp_path):
        """Cycle 10: real indexing via CLI with actual data directory."""
        db_path = str(tmp_path / "chroma_db")

        result = runner.invoke(
            app,
            [
                "index",
                "--data-dir",
                str(DATA_DIR),
                "--db-path",
                db_path,
                "--collection-name",
                "cli_test",
            ],
        )

        assert result.exit_code == 0
        assert "Done!" in result.output
        assert "chunks indexed" in result.output
