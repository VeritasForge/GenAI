import json
from unittest.mock import patch

import pytest

from agent.core.agent import Agent, MaxStepsExceededError
from agent.core.types import AgentResult
from agent.tools.registry import ToolRegistry


def _make_registry() -> ToolRegistry:
    """테스트용 레지스트리 생성."""
    registry = ToolRegistry()
    registry.register(
        name="search_pubmed",
        description="Search PubMed for articles",
        parameters={"query": "str", "max_results": "int (default: 5)"},
        func=lambda query, max_results=5: f"Found 3 articles about {query}",
    )
    registry.register(
        name="fetch_article",
        description="Get article details by PMID",
        parameters={"pmid": "str"},
        func=lambda pmid: f"Article {pmid}: Metformin study details",
    )
    return registry


class TestAgent:
    """Agent Loop 테스트."""

    def test_should_return_direct_answer_without_tool_use(self):
        # Given - LLM이 바로 최종 답변을 반환하는 경우
        agent = Agent(registry=_make_registry())
        llm_response = json.dumps(
            {"action": "finish", "answer": "Metformin is a diabetes drug."}
        )

        with patch("agent.core.agent.call_llm", return_value=llm_response):
            # When
            result = agent.run("What is metformin?")

        # Then
        assert isinstance(result, AgentResult)
        assert result.answer == "Metformin is a diabetes drug."
        assert result.steps == []
        assert result.total_steps == 0

    def test_should_execute_tool_and_return_answer(self):
        # Given - LLM이 Tool을 1회 호출 후 답변하는 시나리오
        agent = Agent(registry=_make_registry())

        tool_call = json.dumps(
            {
                "action": "tool",
                "tool": "search_pubmed",
                "args": {"query": "metformin"},
            }
        )
        final_answer = json.dumps(
            {"action": "finish", "answer": "Found 3 relevant articles."}
        )

        with patch("agent.core.agent.call_llm", side_effect=[tool_call, final_answer]):
            # When
            result = agent.run("Search metformin research")

        # Then
        assert result.answer == "Found 3 relevant articles."
        assert len(result.steps) == 1
        assert result.steps[0].action.tool_name == "search_pubmed"
        assert "Found 3 articles" in result.steps[0].observation
        assert result.total_steps == 1

    def test_should_execute_multiple_tools_before_answer(self):
        # Given - 검색 → 상세조회 → 답변의 멀티스텝 시나리오
        agent = Agent(registry=_make_registry())

        step1 = json.dumps(
            {
                "action": "tool",
                "tool": "search_pubmed",
                "args": {"query": "aspirin"},
            }
        )
        step2 = json.dumps(
            {
                "action": "tool",
                "tool": "fetch_article",
                "args": {"pmid": "12345"},
            }
        )
        final = json.dumps({"action": "finish", "answer": "Aspirin study summary."})

        with patch("agent.core.agent.call_llm", side_effect=[step1, step2, final]):
            # When
            result = agent.run("Tell me about aspirin research")

        # Then
        assert result.answer == "Aspirin study summary."
        assert len(result.steps) == 2
        assert result.total_steps == 2

    def test_should_raise_when_max_steps_exceeded(self):
        # Given - LLM이 계속 Tool만 호출하고 끝내지 않는 경우
        agent = Agent(registry=_make_registry())

        infinite_tool_call = json.dumps(
            {
                "action": "tool",
                "tool": "search_pubmed",
                "args": {"query": "loop"},
            }
        )

        with patch("agent.core.agent.call_llm", return_value=infinite_tool_call):
            # When / Then
            with pytest.raises(MaxStepsExceededError):
                agent.run("infinite loop test", max_steps=3)

    def test_should_handle_tool_execution_error_gracefully(self):
        # Given - Tool 실행 중 에러가 발생하는 경우
        registry = ToolRegistry()
        registry.register(
            name="broken_tool",
            description="A tool that always fails",
            parameters={"query": "str"},
            func=lambda query: (_ for _ in ()).throw(RuntimeError("API timeout")),
        )
        agent = Agent(registry=registry)

        tool_call = json.dumps(
            {
                "action": "tool",
                "tool": "broken_tool",
                "args": {"query": "test"},
            }
        )
        final = json.dumps({"action": "finish", "answer": "Sorry, the tool failed."})

        with patch("agent.core.agent.call_llm", side_effect=[tool_call, final]):
            # When
            result = agent.run("test broken tool")

        # Then
        assert "Sorry" in result.answer
        assert len(result.steps) == 1
        assert "Error" in result.steps[0].observation

    def test_should_handle_unknown_tool_in_response(self):
        # Given - LLM이 등록되지 않은 Tool을 호출하는 경우
        agent = Agent(registry=_make_registry())

        bad_tool_call = json.dumps(
            {
                "action": "tool",
                "tool": "nonexistent_tool",
                "args": {},
            }
        )
        final = json.dumps({"action": "finish", "answer": "Let me try differently."})

        with patch("agent.core.agent.call_llm", side_effect=[bad_tool_call, final]):
            # When
            result = agent.run("test unknown tool")

        # Then
        assert result.answer == "Let me try differently."
        assert "Error" in result.steps[0].observation

    def test_should_fallback_to_finish_when_llm_returns_non_json(self):
        # Given - LLM이 JSON 대신 마크다운으로 직접 답변하는 경우
        agent = Agent(registry=_make_registry())
        markdown_response = "## 메트포르민 최신 연구 동향\n\n메트포르민은..."

        with patch("agent.core.agent.call_llm", return_value=markdown_response):
            # When
            result = agent.run("메트포르민 연구")

        # Then - 자연어 응답이 그대로 answer로 반환됨
        assert result.answer == markdown_response
        assert result.steps == []
        assert result.total_steps == 0

    def test_should_fallback_after_tool_use_when_llm_returns_non_json(self):
        # Given - Tool 호출 후 LLM이 자연어로 최종 답변하는 경우
        agent = Agent(registry=_make_registry())

        tool_call = json.dumps(
            {
                "action": "tool",
                "tool": "search_pubmed",
                "args": {"query": "metformin"},
            }
        )
        natural_answer = "메트포르민에 대한 3개의 논문을 찾았습니다."

        with patch(
            "agent.core.agent.call_llm", side_effect=[tool_call, natural_answer]
        ):
            # When
            result = agent.run("메트포르민 검색")

        # Then
        assert result.answer == natural_answer
        assert len(result.steps) == 1
        assert result.total_steps == 1
