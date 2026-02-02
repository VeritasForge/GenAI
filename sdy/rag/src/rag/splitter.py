"""Text Splitting — Step 2 of the RAG pipeline."""

from __future__ import annotations

from rag.loader import Document


def split_document(
    doc: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Split a single Document into smaller chunks with optional overlap."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )
    if not doc.content.strip():
        raise ValueError("Document content is empty")

    text = doc.content
    stride = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0

    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += stride

    chunk_count = len(chunks)
    return [
        Document(
            content=chunk,
            metadata={
                **doc.metadata,
                "chunk_index": i,
                "chunk_count": chunk_count,
                "chunk_char_count": len(chunk),
            },
        )
        for i, chunk in enumerate(chunks)
    ]


def split_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Split multiple Documents into chunks."""
    result: list[Document] = []
    for doc in docs:
        result.extend(split_document(doc, chunk_size, chunk_overlap))
    return result
