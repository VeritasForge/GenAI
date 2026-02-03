import json
import subprocess
from unittest.mock import patch

import pytest

from agent.llm.client import call_llm, call_llm_json


class TestCallLlm:
    """Claude CLI 기본 호출 테스트."""

    def test_should_return_response_text(self):
        # Given
        mock_result = subprocess.CompletedProcess(
            args=["claude", "-p", "hello"],
            returncode=0,
            stdout="Hello! How can I help?",
            stderr="",
        )

        with patch("agent.llm.client.subprocess.run", return_value=mock_result):
            # When
            result = call_llm("hello")

        # Then
        assert result == "Hello! How can I help?"

    def test_should_pass_system_prompt(self):
        # Given
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="response", stderr=""
        )

        with patch("agent.llm.client.subprocess.run", return_value=mock_result) as m:
            # When
            call_llm("hello", system="You are a doctor.")

        # Then
        args = m.call_args[0][0]
        assert "--append-system-prompt" in args
        system_idx = args.index("--append-system-prompt")
        assert args[system_idx + 1] == "You are a doctor."

    def test_should_raise_when_cli_fails(self):
        # Given
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="CLI error"
        )

        with patch("agent.llm.client.subprocess.run", return_value=mock_result):
            # When / Then
            with pytest.raises(RuntimeError, match="Claude CLI"):
                call_llm("hello")

    def test_should_raise_when_prompt_is_empty(self):
        # When / Then
        with pytest.raises(ValueError, match="prompt"):
            call_llm("")


class TestCallLlmJson:
    """JSON 응답 파싱 테스트."""

    def test_should_parse_json_response(self):
        # Given
        json_str = json.dumps({"action": "finish", "answer": "42"})
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json_str, stderr=""
        )

        with patch("agent.llm.client.subprocess.run", return_value=mock_result):
            # When
            result = call_llm_json("What is 6*7?")

        # Then
        assert result == {"action": "finish", "answer": "42"}

    def test_should_extract_json_from_markdown_code_block(self):
        # Given - LLM이 ```json ... ``` 블록으로 감싸서 응답하는 경우
        response = '```json\n{"action": "tool", "tool": "search"}\n```'
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=response, stderr=""
        )

        with patch("agent.llm.client.subprocess.run", return_value=mock_result):
            # When
            result = call_llm_json("query")

        # Then
        assert result == {"action": "tool", "tool": "search"}

    def test_should_raise_when_response_is_not_json(self):
        # Given
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="This is not JSON", stderr=""
        )

        with patch("agent.llm.client.subprocess.run", return_value=mock_result):
            # When / Then
            with pytest.raises(ValueError, match="JSON"):
                call_llm_json("query")
