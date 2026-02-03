import pytest

from agent.core.parser import parse_llm_response
from agent.core.types import AgentAction, AgentFinish


class TestParseLlmResponse:
    """LLM 응답 파싱 테스트."""

    def test_should_parse_tool_action(self):
        # Given
        response = (
            '{"action": "tool", "tool": "search_pubmed",'
            ' "args": {"query": "metformin"}}'
        )

        # When
        result = parse_llm_response(response)

        # Then
        assert isinstance(result, AgentAction)
        assert result.tool_name == "search_pubmed"
        assert result.tool_args == {"query": "metformin"}

    def test_should_parse_finish_action(self):
        # Given
        response = '{"action": "finish", "answer": "Metformin is a drug..."}'

        # When
        result = parse_llm_response(response)

        # Then
        assert isinstance(result, AgentFinish)
        assert result.answer == "Metformin is a drug..."

    def test_should_handle_json_in_code_block(self):
        # Given
        response = (
            '```json\n{"action": "tool",'
            ' "tool": "fetch_article", "args": {"pmid": "123"}}\n```'
        )

        # When
        result = parse_llm_response(response)

        # Then
        assert isinstance(result, AgentAction)
        assert result.tool_name == "fetch_article"

    def test_should_raise_on_invalid_json(self):
        # Given
        response = "I think we should search PubMed"

        # When / Then
        with pytest.raises(ValueError, match="JSON"):
            parse_llm_response(response)

    def test_should_raise_on_unknown_action(self):
        # Given
        response = '{"action": "unknown", "data": "something"}'

        # When / Then
        with pytest.raises(ValueError, match="action"):
            parse_llm_response(response)

    def test_should_raise_when_tool_action_missing_tool_name(self):
        # Given
        response = '{"action": "tool", "args": {"query": "test"}}'

        # When / Then
        with pytest.raises(ValueError, match="tool"):
            parse_llm_response(response)

    def test_should_raise_when_finish_action_missing_answer(self):
        # Given
        response = '{"action": "finish"}'

        # When / Then
        with pytest.raises(ValueError, match="answer"):
            parse_llm_response(response)

    def test_should_default_args_to_empty_dict(self):
        # Given - args가 없는 tool 호출
        response = '{"action": "tool", "tool": "list_tools"}'

        # When
        result = parse_llm_response(response)

        # Then
        assert isinstance(result, AgentAction)
        assert result.tool_args == {}
