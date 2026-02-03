"""LLM 응답 파서 — JSON 응답에서 AgentAction 또는 AgentFinish를 추출한다."""

import json
import re

from agent.core.types import AgentAction, AgentFinish


def parse_llm_response(response: str) -> AgentAction | AgentFinish:
    """LLM 응답 텍스트를 파싱하여 AgentAction 또는 AgentFinish를 반환한다.

    Args:
        response: LLM의 원시 텍스트 응답.

    Returns:
        AgentAction (Tool 호출) 또는 AgentFinish (최종 답변).

    Raises:
        ValueError: JSON 파싱 실패, 알 수 없는 action, 필수 필드 누락.
    """
    data = _extract_json(response)
    action = data.get("action")

    if action == "tool":
        tool_name = data.get("tool")
        if not tool_name:
            raise ValueError("'tool' action requires 'tool' field")
        return AgentAction(
            tool_name=tool_name,
            tool_args=data.get("args", {}),
        )

    if action == "finish":
        answer = data.get("answer")
        if answer is None:
            raise ValueError("'finish' action requires 'answer' field")
        return AgentFinish(answer=answer)

    raise ValueError(f"Unknown action: {action}")


def _extract_json(text: str) -> dict:
    """텍스트에서 JSON 객체를 추출한다."""
    # 1차: 직접 파싱
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2차: ```json ... ``` 블록에서 추출
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse JSON from response: {text[:200]}")
