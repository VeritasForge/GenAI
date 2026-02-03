"""TDD tests for ClaudeCodeLLM (LlamaIndex CustomLLM)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.llms import CompletionResponse, LLMMetadata

from rag_llamaindex.llm import ClaudeCodeLLM


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


# --- Group A: Metadata ---


class TestMetadata:
    def test_should_return_llm_metadata(self):
        """Cycle 1: metadata returns LLMMetadata with model_name."""
        # Given
        llm = ClaudeCodeLLM()

        # When
        meta = llm.metadata

        # Then
        assert isinstance(meta, LLMMetadata)
        assert meta.model_name == "claude-code"
        assert meta.is_chat_model is False

    def test_should_have_class_name(self):
        """Cycle 2: class_name returns 'ClaudeCodeLLM'."""
        assert ClaudeCodeLLM.class_name() == "ClaudeCodeLLM"


# --- Group B: Completion ---


class TestCompletion:
    @patch("rag_llamaindex.llm.subprocess.run")
    def test_should_return_completion_response(self, mock_run):
        """Cycle 3: complete() returns CompletionResponse with text."""
        # Given
        mock_run.return_value = _mock_subprocess_success("Hello from Claude")
        llm = ClaudeCodeLLM()

        # When
        response = llm.complete("Say hello")

        # Then
        assert isinstance(response, CompletionResponse)
        assert response.text == "Hello from Claude"

    @patch("rag_llamaindex.llm.subprocess.run")
    def test_should_call_claude_cli_with_correct_args(self, mock_run):
        """Cycle 4: subprocess.run called with ['claude', '-p', prompt]."""
        # Given
        mock_run.return_value = _mock_subprocess_success()
        llm = ClaudeCodeLLM()

        # When
        llm.complete("test prompt")

        # Then
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "claude"
        assert cmd[1] == "-p"
        assert cmd[2] == "test prompt"
        assert call_args[1]["capture_output"] is True
        assert call_args[1]["text"] is True
        assert call_args[1]["timeout"] == 120


# --- Group C: Stream Completion ---


class TestStreamCompletion:
    @patch("rag_llamaindex.llm.subprocess.run")
    def test_should_yield_single_response(self, mock_run):
        """Cycle 5: stream_complete() yields single CompletionResponse."""
        # Given
        mock_run.return_value = _mock_subprocess_success("Streamed answer")
        llm = ClaudeCodeLLM()

        # When
        responses = list(llm.stream_complete("test"))

        # Then
        assert len(responses) == 1
        assert responses[0].text == "Streamed answer"
        assert responses[0].delta == "Streamed answer"


# --- Group D: Error Handling ---


class TestErrorHandling:
    @patch("rag_llamaindex.llm.subprocess.run")
    def test_should_raise_on_cli_failure(self, mock_run):
        """Cycle 6: returncode != 0 raises RuntimeError."""
        # Given
        mock_run.return_value = _mock_subprocess_failure("something went wrong")
        llm = ClaudeCodeLLM()

        # When / Then
        with pytest.raises(RuntimeError, match="Claude Code CLI failed"):
            llm.complete("test")

    @patch("rag_llamaindex.llm.subprocess.run")
    def test_should_raise_on_timeout(self, mock_run):
        """Cycle 7: subprocess timeout propagates."""
        # Given
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)
        llm = ClaudeCodeLLM()

        # When / Then
        with pytest.raises(subprocess.TimeoutExpired):
            llm.complete("test")

    @patch("rag_llamaindex.llm.subprocess.run")
    def test_should_strip_whitespace_from_output(self, mock_run):
        """Cycle 8: output is stripped of leading/trailing whitespace."""
        # Given
        mock_run.return_value = _mock_subprocess_success("  answer with spaces  \n")
        llm = ClaudeCodeLLM()

        # When
        response = llm.complete("test")

        # Then
        assert response.text == "answer with spaces"
