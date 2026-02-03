"""Ingestion module — load, split, and store documents (LangChain)."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Create HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )


def load_and_split(
    data_dir: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Load .txt files from directory and split into chunks.

    Args:
        data_dir: Directory containing .txt files.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of Document chunks with metadata.

    Raises:
        ValueError: If directory doesn't exist or has no .txt files.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise ValueError(f"Directory does not exist: {data_dir}")

    txt_files = list(data_dir.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {data_dir}")

    loader = DirectoryLoader(
        str(data_dir),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def create_vector_store(
    documents: list[Document] | None = None,
    db_path: str = "./chroma_db",
    collection_name: str = "rag",
) -> Chroma:
    """Create or load a Chroma vector store.

    Args:
        documents: Documents to add. None to load existing store.
        db_path: ChromaDB persist directory.
        collection_name: ChromaDB collection name.

    Returns:
        Chroma vector store ready for querying.
    """
    embeddings = _get_embeddings()

    if documents is not None:
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=db_path,
        )

    return Chroma(
        collection_name=collection_name,
        persist_directory=db_path,
        embedding_function=embeddings,
    )
