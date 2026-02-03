"""에이전트 코어 타입 정의."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentAction:
    """에이전트가 Tool을 호출하려는 의도."""

    tool_name: str
    tool_args: dict[str, Any]


@dataclass(frozen=True)
class AgentFinish:
    """에이전트가 최종 답변을 반환하려는 의도."""

    answer: str


@dataclass(frozen=True)
class AgentStep:
    """에이전트 루프의 한 단계 기록."""

    action: AgentAction
    observation: str


@dataclass(frozen=True)
class AgentResult:
    """에이전트 실행의 최종 결과."""

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_steps: int = 0
