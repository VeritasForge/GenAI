"""Generation — Step 6 of the RAG pipeline."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from rag.store import SearchResult


@dataclass
class GenerationResult:
    """Container for LLM generation output."""

    answer: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


def format_context(results: list[SearchResult]) -> str:
    """Format search results into a numbered context string.

    Each result becomes:
        [N] content
        (source: filename, chunk M, score: 0.xx)
    """
    if not results:
        return "(No relevant documents found.)"

    lines: list[str] = []
    for i, result in enumerate(results, start=1):
        meta = result.document.metadata
        filename = meta.get("filename", "unknown")
        chunk_index = meta.get("chunk_index", "?")
        score = result.score

        lines.append(f"[{i}] {result.document.content}")
        lines.append(f"(source: {filename}, chunk {chunk_index}, score: {score:.2f})")
        lines.append("")

    return "\n".join(lines).strip()


def build_prompt(query: str, context: str, system_prompt: str) -> str:
    """Combine system prompt, context, and question into a single prompt string."""
    return f"""{system_prompt}

Context:
---
{context}
---

Question: {query}"""


class Generator:
    """Generate answers using Claude Code CLI as LLM backend."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant. "
        "Answer the question based ONLY on the provided context. "
        "If the context doesn't contain relevant information, say so."
    )

    def __init__(
        self,
        system_prompt: str | None = None,
    ) -> None:
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def generate(
        self,
        query: str,
        results: list[SearchResult],
    ) -> GenerationResult:
        """Generate an answer from search results using Claude Code CLI."""
        context = format_context(results)
        prompt = build_prompt(query, context, self._system_prompt)
        answer = self._call_claude(prompt)
        return GenerationResult(answer=answer, model="claude-code")

    def _call_claude(self, prompt: str) -> str:
        """Call Claude Code CLI via subprocess."""
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Claude Code CLI failed: {result.stderr.strip()}"
            )
        return result.stdout.strip()
