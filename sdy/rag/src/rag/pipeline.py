"""Pipeline orchestration — Step 7 of the RAG pipeline.

Pure logic layer: no CLI dependency, easy to test and reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rag.embedder import embed_documents
from rag.generator import Generator
from rag.loader import load_directory
from rag.retriever import Retriever
from rag.splitter import split_documents
from rag.store import SearchResult, VectorStore


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
    sources: list[SearchResult] = field(default_factory=list)
    model: str = "claude-code"


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
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise ValueError(f"Directory does not exist: {data_dir}")

    txt_files = list(data_dir.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {data_dir}")

    docs = load_directory(data_dir)
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedded = embed_documents(chunks)

    store = VectorStore(
        collection_name=collection_name,
        persist_directory=db_path,
    )
    store.add_documents(embedded)

    return IndexResult(
        total_documents=len(docs),
        total_chunks=len(chunks),
        db_path=db_path,
        collection_name=collection_name,
    )


def ask_question(
    query: str,
    db_path: str = "./chroma_db",
    collection_name: str = "rag",
    top_k: int = 5,
    score_threshold: float = 0.0,
    system_prompt: str | None = None,
) -> AskResult:
    """Retrieve -> Generate.

    Args:
        query: Natural-language question.
        db_path: ChromaDB persist directory.
        collection_name: ChromaDB collection name.
        top_k: Maximum number of results.
        score_threshold: Minimum similarity score.
        system_prompt: Optional custom system prompt.

    Returns:
        AskResult with answer and sources.

    Raises:
        ValueError: If query is empty or whitespace.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    store = VectorStore(
        collection_name=collection_name,
        persist_directory=db_path,
    )
    retriever = Retriever(
        store=store,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    generator = Generator(system_prompt=system_prompt)

    results = retriever.retrieve(query)
    generation = generator.generate(query, results)

    return AskResult(
        answer=generation.answer,
        sources=results,
        model=generation.model,
    )
