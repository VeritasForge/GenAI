"""Pipeline orchestration — load, index, and query (LangChain).

Pure logic layer: no CLI dependency, easy to test and reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rag_langchain.chain import QueryResult, SourceInfo, query_with_chain
from rag_langchain.ingestion import create_vector_store, load_and_split


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
    docs = load_and_split(
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Count unique source files
    source_files = {doc.metadata.get("source", "") for doc in docs}

    create_vector_store(
        documents=docs,
        db_path=db_path,
        collection_name=collection_name,
    )

    return IndexResult(
        total_documents=len(source_files),
        total_chunks=len(docs),
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

    store = create_vector_store(
        documents=None,
        db_path=db_path,
        collection_name=collection_name,
    )

    result: QueryResult = query_with_chain(store, query, top_k=top_k)

    return AskResult(
        answer=result.answer,
        sources=result.sources,
    )
