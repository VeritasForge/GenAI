"""Pipeline orchestration — load, index, and query (LlamaIndex).

Pure logic layer: no CLI dependency, easy to test and reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rag_llamaindex.ingestion import create_index, load_and_split
from rag_llamaindex.query import QueryResult, SourceInfo, query_index


@dataclass
class IndexResult:
    """Result of an indexing operation."""

    total_documents: int
    total_chunks: int
    db_path: str
    collection_name: str


@dataclass
class AskResult:
    """Result of a question-answering operation."""

    answer: str
    sources: list[SourceInfo] = field(default_factory=list)


def index_documents(
    data_dir: str | Path,
    db_path: str = "./chroma_db",
    collection_name: str = "rag",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> IndexResult:
    """Load -> Split -> Embed -> Store.

    Args:
        data_dir: Directory containing .txt files.
        db_path: ChromaDB persist directory.
        collection_name: ChromaDB collection name.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        IndexResult with counts and paths.

    Raises:
        ValueError: If data_dir doesn't exist or has no .txt files.
    """
    nodes = load_and_split(
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Count unique documents by file_path metadata
    doc_files = {
        node.metadata.get("file_path", node.metadata.get("file_name", ""))
        for node in nodes
    }

    create_index(
        nodes=nodes,
        db_path=db_path,
        collection_name=collection_name,
    )

    return IndexResult(
        total_documents=len(doc_files),
        total_chunks=len(nodes),
        db_path=db_path,
        collection_name=collection_name,
    )


def ask_question(
    query: str,
    db_path: str = "./chroma_db",
    collection_name: str = "rag",
    top_k: int = 5,
) -> AskResult:
    """Retrieve -> Generate.

    Args:
        query: Natural-language question.
        db_path: ChromaDB persist directory.
        collection_name: ChromaDB collection name.
        top_k: Maximum number of results.

    Returns:
        AskResult with answer and sources.

    Raises:
        ValueError: If query is empty or whitespace.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    index = create_index(
        nodes=None,
        db_path=db_path,
        collection_name=collection_name,
    )

    result: QueryResult = query_index(index, query, top_k=top_k)

    return AskResult(
        answer=result.answer,
        sources=result.sources,
    )
