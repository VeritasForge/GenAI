"""Agent Loop — ReAct 패턴 구현.

Observe → Think → Act 루프를 반복하여 복잡한 질문에 답한다.
LLM이 다음 행동을 동적으로 결정하는 것이 핵심.
"""

import json

from agent.core.parser import parse_llm_response
from agent.core.types import AgentAction, AgentFinish, AgentResult, AgentStep
from agent.llm.client import call_llm
from agent.llm.prompts import AGENT_SYSTEM_PROMPT
from agent.tools.registry import ToolRegistry


class MaxStepsExceededError(Exception):
    """에이전트가 최대 스텝 수를 초과한 경우."""


class Agent:
    """ReAct 패턴 기반 멀티스텝 에이전트."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def run(self, query: str, max_steps: int = 10) -> AgentResult:
        """사용자 질문에 대해 Tool을 활용하여 답변을 생성한다.

        Args:
            query: 사용자 질문.
            max_steps: 최대 루프 반복 횟수.

        Returns:
            최종 답변과 실행 과정이 담긴 AgentResult.

        Raises:
            MaxStepsExceededError: max_steps를 초과한 경우.
        """
        system = AGENT_SYSTEM_PROMPT.format(
            tool_descriptions=self._registry.get_prompt_description()
        )
        messages = [f"User: {query}"]
        steps: list[AgentStep] = []

        for _ in range(max_steps):
            prompt = "\n".join(messages)
            raw_response = call_llm(prompt, system=system)

            try:
                parsed = parse_llm_response(raw_response)
            except ValueError:
                parsed = AgentFinish(answer=raw_response)

            if isinstance(parsed, AgentFinish):
                return AgentResult(
                    answer=parsed.answer,
                    steps=steps,
                    total_steps=len(steps),
                )

            observation = self._execute_tool(parsed)
            steps.append(AgentStep(action=parsed, observation=observation))

            messages.append(f"Assistant: {raw_response}")
            messages.append(f"Observation: {observation}")

        raise MaxStepsExceededError(
            f"Agent exceeded {max_steps} steps without finishing"
        )

    def _execute_tool(self, action: AgentAction) -> str:
        """Tool을 실행하고 결과를 문자열로 반환한다."""
        try:
            result = self._registry.execute(action.tool_name, action.tool_args)
            return self._format_observation(result)
        except Exception as e:
            return f"Error executing {action.tool_name}: {e}"

    def _format_observation(self, result: object) -> str:
        """Tool 실행 결과를 문자열로 변환한다."""
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            return json.dumps(
                [self._to_serializable(item) for item in result],
                ensure_ascii=False,
                indent=2,
            )
        return str(result)

    def _to_serializable(self, obj: object) -> object:
        """dataclass 등을 직렬화 가능한 형태로 변환한다."""
        if hasattr(obj, "__dataclass_fields__"):
            from dataclasses import asdict

            return asdict(obj)
        return obj
