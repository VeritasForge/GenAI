"""Tool Registry — 에이전트가 사용할 수 있는 Tool을 등록하고 관리한다."""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolSpec:
    """Tool의 메타데이터와 실행 함수."""

    name: str
    description: str
    parameters: dict[str, str]
    func: Callable[..., Any]


class ToolRegistry:
    """에이전트가 사용 가능한 Tool 목록을 관리한다."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, str],
        func: Callable[..., Any],
    ) -> None:
        """Tool을 레지스트리에 등록한다.

        Raises:
            ValueError: 이미 등록된 이름인 경우.
        """
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
        )

    def execute(self, name: str, args: dict[str, Any]) -> Any:
        """등록된 Tool을 실행한다.

        Raises:
            KeyError: 등록되지 않은 Tool인 경우.
        """
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].func(**args)

    def get_prompt_description(self) -> str:
        """LLM 프롬프트에 삽입할 Tool 목록 텍스트를 생성한다."""
        lines = []
        for i, spec in enumerate(self._tools.values(), 1):
            params_str = ", ".join(f'"{k}": "{v}"' for k, v in spec.parameters.items())
            lines.append(f"{i}. {spec.name}")
            lines.append(f"   Description: {spec.description}")
            lines.append(f"   Parameters: {{{params_str}}}")
            lines.append("")
        return "\n".join(lines)
