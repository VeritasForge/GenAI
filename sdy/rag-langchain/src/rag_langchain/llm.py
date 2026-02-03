"""Custom LLM using Claude Code CLI as backend (LangChain)."""

from __future__ import annotations

import subprocess
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM


class ClaudeCodeLLM(LLM):
    """LangChain LLM that calls Claude Code CLI via subprocess."""

    model_name: str = "claude-code"

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    @property
    def _identifying_params(self) -> dict[str, str]:
        return {"model_name": self.model_name}

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Call Claude Code CLI and return the response text."""
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Claude Code CLI failed: {result.stderr.strip()}")
        return result.stdout.strip()
