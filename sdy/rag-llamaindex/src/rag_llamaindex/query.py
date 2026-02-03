"""Query module — query engine creation and response formatting (LlamaIndex)."""

from __future__ import annotations

from dataclasses import dataclass, field

from llama_index.core import VectorStoreIndex

from rag_llamaindex.llm import ClaudeCodeLLM


@dataclass
class SourceInfo:
    """Simplified source information from a retrieval result."""

    text: str
    metadata: dict[str, str | int]
    score: float | None = None


@dataclass
class QueryResult:
    """Result of a query operation."""

    answer: str
    sources: list[SourceInfo] = field(default_factory=list)


def query_index(
    index: VectorStoreIndex,
    query: str,
    top_k: int = 5,
) -> QueryResult:
    """Query the index and return formatted results.

    Args:
        index: VectorStoreIndex to query.
        query: Natural-language question.
        top_k: Maximum number of source nodes.

    Returns:
        QueryResult with answer and source information.

    Raises:
        ValueError: If query is empty or whitespace.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    llm = ClaudeCodeLLM()
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=top_k,
    )
    response = query_engine.query(query)

    sources = []
    for node_with_score in response.source_nodes:
        sources.append(
            SourceInfo(
                text=node_with_score.node.get_content(),
                metadata=dict(node_with_score.node.metadata),
                score=node_with_score.score,
            )
        )

    return QueryResult(
        answer=str(response).strip(),
        sources=sources,
    )
