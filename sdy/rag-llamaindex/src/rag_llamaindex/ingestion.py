"""Ingestion module — load, split, and index documents (LlamaIndex)."""

from __future__ import annotations

from pathlib import Path

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


def load_and_split(
    data_dir: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[BaseNode]:
    """Load .txt files from directory and split into nodes.

    Args:
        data_dir: Directory containing .txt files.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of nodes (chunks) with metadata.

    Raises:
        ValueError: If directory doesn't exist or has no .txt files.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise ValueError(f"Directory does not exist: {data_dir}")

    txt_files = list(data_dir.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {data_dir}")

    reader = SimpleDirectoryReader(
        input_dir=str(data_dir),
        required_exts=[".txt"],
    )
    documents = reader.load_data()

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.get_nodes_from_documents(documents)


def create_index(
    nodes: list[BaseNode] | None = None,
    db_path: str = "./chroma_db",
    collection_name: str = "rag",
) -> VectorStoreIndex:
    """Create or load a VectorStoreIndex backed by ChromaDB.

    Args:
        nodes: Nodes to add. None to load existing index.
        db_path: ChromaDB persist directory.
        collection_name: ChromaDB collection name.

    Returns:
        VectorStoreIndex ready for querying.
    """
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

    if nodes is not None:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model,
        )

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
