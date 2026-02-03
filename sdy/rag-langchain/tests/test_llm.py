"""TDD tests for ClaudeCodeLLM (LangChain LLM)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_langchain.llm import ClaudeCodeLLM


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
    def test_should_have_llm_type(self):
        """Cycle 1: _llm_type returns 'claude-code'."""
        # Given
        llm = ClaudeCodeLLM()

        # When / Then
        assert llm._llm_type == "claude-code"

    def test_should_have_identifying_params(self):
        """Cycle 2: _identifying_params contains model_name."""
        # Given
        llm = ClaudeCodeLLM()

        # When
        params = llm._identifying_params

        # Then
        assert "model_name" in params
        assert params["model_name"] == "claude-code"


# --- Group B: Call ---


class TestCall:
    @patch("rag_langchain.llm.subprocess.run")
    def test_should_return_string(self, mock_run):
        """Cycle 3: invoke() returns string response."""
        # Given
        mock_run.return_value = _mock_subprocess_success("Hello from Claude")
        llm = ClaudeCodeLLM()

        # When
        response = llm.invoke("Say hello")

        # Then
        assert response == "Hello from Claude"

    @patch("rag_langchain.llm.subprocess.run")
    def test_should_call_claude_cli_with_correct_args(self, mock_run):
        """Cycle 4: subprocess.run called with ['claude', '-p', prompt]."""
        # Given
        mock_run.return_value = _mock_subprocess_success()
        llm = ClaudeCodeLLM()

        # When
        llm.invoke("test prompt")

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

    @patch("rag_langchain.llm.subprocess.run")
    def test_should_batch_multiple_prompts(self, mock_run):
        """Cycle 5: generate() handles multiple prompts."""
        # Given
        mock_run.return_value = _mock_subprocess_success("Answer")
        llm = ClaudeCodeLLM()

        # When
        result = llm.generate(["prompt 1", "prompt 2"])

        # Then
        assert len(result.generations) == 2
        assert result.generations[0][0].text == "Answer"
        assert result.generations[1][0].text == "Answer"


# --- Group C: Error Handling ---


class TestErrorHandling:
    @patch("rag_langchain.llm.subprocess.run")
    def test_should_raise_on_cli_failure(self, mock_run):
        """Cycle 6: returncode != 0 raises RuntimeError."""
        # Given
        mock_run.return_value = _mock_subprocess_failure("something went wrong")
        llm = ClaudeCodeLLM()

        # When / Then
        with pytest.raises(RuntimeError, match="Claude Code CLI failed"):
            llm.invoke("test")

    @patch("rag_langchain.llm.subprocess.run")
    def test_should_raise_on_timeout(self, mock_run):
        """Cycle 7: subprocess timeout propagates."""
        # Given
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)
        llm = ClaudeCodeLLM()

        # When / Then
        with pytest.raises(subprocess.TimeoutExpired):
            llm.invoke("test")

    @patch("rag_langchain.llm.subprocess.run")
    def test_should_strip_whitespace_from_output(self, mock_run):
        """Cycle 8: output is stripped of leading/trailing whitespace."""
        # Given
        mock_run.return_value = _mock_subprocess_success("  answer with spaces  \n")
        llm = ClaudeCodeLLM()

        # When
        response = llm.invoke("test")

        # Then
        assert response == "answer with spaces"
