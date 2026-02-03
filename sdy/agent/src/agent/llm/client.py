"""Claude CLI 래퍼.

`claude -p` 명령을 subprocess로 호출하여 LLM 응답을 받는다.
JSON 응답 파싱을 지원하여 에이전트의 구조화된 출력을 처리한다.
"""

import json
import re
import subprocess


def call_llm(prompt: str, system: str | None = None) -> str:
    """Claude CLI를 호출하여 텍스트 응답을 반환한다.

    Args:
        prompt: 사용자 프롬프트.
        system: 시스템 프롬프트 (선택).

    Returns:
        LLM 응답 텍스트.

    Raises:
        ValueError: prompt가 비어있는 경우.
        RuntimeError: CLI 호출 실패 시.
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    cmd = ["claude", "-p", prompt]
    if system:
        cmd.extend(["--append-system-prompt", system])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI error: {result.stderr}")

    return result.stdout.strip()


def call_llm_json(prompt: str, system: str | None = None) -> dict:
    """Claude CLI를 호출하여 JSON 응답을 파싱하여 반환한다.

    LLM이 ```json ... ``` 블록으로 감싸서 응답하는 경우도 처리한다.

    Args:
        prompt: 사용자 프롬프트.
        system: 시스템 프롬프트 (선택).

    Returns:
        파싱된 JSON 딕셔너리.

    Raises:
        ValueError: 응답이 유효한 JSON이 아닌 경우.
    """
    raw = call_llm(prompt, system)
    return _parse_json(raw)


def _parse_json(text: str) -> dict:
    """텍스트에서 JSON을 추출하여 파싱한다."""
    # 1차: 직접 파싱 시도
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2차: ```json ... ``` 블록에서 추출
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse JSON from LLM response: {text[:200]}")
