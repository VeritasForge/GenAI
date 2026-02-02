"""TDD tests for Generator (Step 6)."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.generator import GenerationResult, Generator, build_prompt, format_context
from rag.loader import Document
from rag.store import SearchResult


def _make_result(
    content: str,
    score: float,
    filename: str = "test.txt",
    chunk_index: int = 0,
) -> SearchResult:
    """Helper to create a SearchResult for testing."""
    return SearchResult(
        document=Document(
            content=content,
            metadata={"filename": filename, "chunk_index": chunk_index},
        ),
        score=score,
    )


def _mock_subprocess_success(answer: str = "Mock answer") -> MagicMock:
    """Create a mock subprocess.run return value for success."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = answer
    mock_result.stderr = ""
    return mock_result


def _mock_subprocess_failure(stderr: str = "error occurred") -> MagicMock:
    """Create a mock subprocess.run return value for failure."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = stderr
    return mock_result


# --- Group A: 기본 동작 ---


class TestBasicBehavior:
    def test_format_context_basic(self):
        """Cycle 1: SearchResult → 번호 매긴 문자열."""
        results = [_make_result("metformin treats diabetes", 0.85)]
        context = format_context(results)

        assert "[1]" in context
        assert "metformin treats diabetes" in context

    def test_build_prompt_structure(self):
        """Cycle 2: system + context + question 조합."""
        prompt = build_prompt(
            query="What is metformin?",
            context="[1] metformin is a drug",
            system_prompt="You are helpful.",
        )

        assert "You are helpful." in prompt
        assert "[1] metformin is a drug" in prompt
        assert "Question: What is metformin?" in prompt
        assert "Context:" in prompt

    @patch("rag.generator.subprocess.run")
    def test_generate_returns_result(self, mock_run):
        """Cycle 3: mock subprocess → GenerationResult 반환."""
        answer = "Metformin is used for diabetes."
        mock_run.return_value = _mock_subprocess_success(answer)

        gen = Generator()
        result = gen.generate(
            query="What is metformin?",
            results=[_make_result("metformin treats diabetes", 0.85)],
        )

        assert isinstance(result, GenerationResult)
        assert result.answer == "Metformin is used for diabetes."
        assert result.model == "claude-code"
        assert result.prompt_tokens is None
        assert result.completion_tokens is None

    @patch("rag.generator.subprocess.run")
    def test_generate_calls_claude_cli(self, mock_run):
        """Cycle 4: subprocess.run 호출 인자 검증."""
        mock_run.return_value = _mock_subprocess_success()

        gen = Generator()
        gen.generate(
            query="test question",
            results=[_make_result("some content", 0.5)],
        )

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]

        assert cmd[0] == "claude"
        assert cmd[1] == "-p"
        assert call_args[1]["capture_output"] is True
        assert call_args[1]["text"] is True
        assert call_args[1]["timeout"] == 120


# --- Group B: Context 포맷팅 ---


class TestContextFormatting:
    def test_format_context_multiple_results(self):
        """Cycle 5: 여러 결과 → [1]...[2]...[3]..."""
        results = [
            _make_result("first doc", 0.9, "a.txt", 0),
            _make_result("second doc", 0.8, "b.txt", 1),
            _make_result("third doc", 0.7, "c.txt", 2),
        ]
        context = format_context(results)

        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context
        assert "first doc" in context
        assert "second doc" in context
        assert "third doc" in context

    def test_format_context_includes_metadata(self):
        """Cycle 6: filename, chunk_index, score 포함."""
        results = [_make_result("content", 0.75, "metformin.txt", 3)]
        context = format_context(results)

        assert "metformin.txt" in context
        assert "chunk 3" in context
        assert "0.75" in context

    def test_format_context_missing_metadata(self):
        """Cycle 7: metadata 키 없어도 에러 안 남."""
        result = SearchResult(
            document=Document(content="bare content", metadata={}),
            score=0.5,
        )
        context = format_context([result])

        assert "bare content" in context
        assert "unknown" in context  # filename fallback
        assert "?" in context  # chunk_index fallback


# --- Group C: 에지 케이스 ---


class TestEdgeCases:
    def test_format_context_empty_results(self):
        """Cycle 8: 빈 리스트 → "(No relevant documents found.)"."""
        context = format_context([])
        assert context == "(No relevant documents found.)"

    @patch("rag.generator.subprocess.run")
    def test_generate_with_empty_results(self, mock_run):
        """Cycle 9: 빈 context로도 CLI 호출 수행."""
        mock_run.return_value = _mock_subprocess_success("No information available.")

        gen = Generator()
        result = gen.generate(query="anything", results=[])

        assert result.answer == "No information available."
        mock_run.assert_called_once()

    @patch("rag.generator.subprocess.run")
    def test_generate_cli_error_propagates(self, mock_run):
        """Cycle 10: returncode != 0 → RuntimeError."""
        mock_run.return_value = _mock_subprocess_failure("something went wrong")

        gen = Generator()
        with pytest.raises(RuntimeError, match="Claude Code CLI failed"):
            gen.generate(
                query="test",
                results=[_make_result("content", 0.5)],
            )

    @patch("rag.generator.subprocess.run")
    def test_custom_system_prompt(self, mock_run):
        """Cycle 11: 커스텀 system prompt 사용."""
        mock_run.return_value = _mock_subprocess_success("custom answer")

        custom_prompt = "You are a medical expert."
        gen = Generator(system_prompt=custom_prompt)
        gen.generate(
            query="test",
            results=[_make_result("content", 0.5)],
        )

        call_args = mock_run.call_args
        prompt_arg = call_args[0][0][2]  # claude -p <prompt>
        assert "You are a medical expert." in prompt_arg


# --- Group D: 통합 ---


DATA_DIR = Path(__file__).parent.parent / "data"


class TestIntegration:
    @pytest.mark.skipif(
        shutil.which("claude") is None,
        reason="claude CLI not found in PATH",
    )
    def test_full_pipeline_generate(self):
        """Cycle 12: load→split→embed→store→retrieve→generate (claude CLI 필요)."""
        from rag.embedder import embed_documents
        from rag.loader import load_file
        from rag.retriever import Retriever
        from rag.splitter import split_document
        from rag.store import VectorStore

        doc = load_file(DATA_DIR / "metformin_overview.txt")
        chunks = split_document(doc, chunk_size=300, chunk_overlap=30)
        embedded = embed_documents(chunks)

        store = VectorStore(collection_name="test_gen_pipeline")
        store.add_documents(embedded)

        retriever = Retriever(store=store, top_k=3, score_threshold=0.1)
        results = retriever.retrieve("what are the side effects of metformin?")

        gen = Generator()
        gen_result = gen.generate(
            query="what are the side effects of metformin?",
            results=results,
        )

        assert isinstance(gen_result, GenerationResult)
        assert len(gen_result.answer) > 0
        assert gen_result.model == "claude-code"
