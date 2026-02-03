"""Custom LLM using Claude Code CLI as backend (LlamaIndex)."""

from __future__ import annotations

import subprocess
from typing import Any

from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback


class ClaudeCodeLLM(CustomLLM):
    """LlamaIndex CustomLLM that calls Claude Code CLI via subprocess."""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=200000,
            num_output=4096,
            is_chat_model=False,
            model_name="claude-code",
        )

    @classmethod
    def class_name(cls) -> str:
        return "ClaudeCodeLLM"

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        text = self._call_claude(prompt)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        text = self._call_claude(prompt)
        yield CompletionResponse(text=text, delta=text)

    def _call_claude(self, prompt: str) -> str:
        """Call Claude Code CLI via subprocess."""
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Claude Code CLI failed: {result.stderr.strip()}")
        return result.stdout.strip()
