import json
from unittest.mock import patch

from agent.core.types import AgentResult
from agent.pipeline import ask_question, create_default_registry


class TestCreateDefaultRegistry:
    """기본 레지스트리 생성 테스트."""

    def test_should_include_pubmed_tools(self):
        # When
        registry = create_default_registry()

        # Then
        assert "search_pubmed" in registry.tool_names
        assert "fetch_article" in registry.tool_names

    def test_should_include_clinical_trials_tool(self):
        # When
        registry = create_default_registry()

        # Then
        assert "search_trials" in registry.tool_names


class TestAskQuestion:
    """단일 질문 에이전트 테스트."""

    def test_should_return_agent_result(self):
        # Given
        finish_response = json.dumps(
            {"action": "finish", "answer": "Metformin is a diabetes drug."}
        )

        with patch("agent.core.agent.call_llm", return_value=finish_response):
            # When
            result = ask_question("What is metformin?")

        # Then
        assert isinstance(result, AgentResult)
        assert "Metformin" in result.answer
