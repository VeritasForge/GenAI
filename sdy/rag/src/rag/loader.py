"""Document Loading — Step 1 of the RAG pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """A loaded document with content and metadata."""

    content: str
    metadata: dict[str, str | int] = field(default_factory=dict)


def load_file(path: str | Path) -> Document:
    """Load a single .txt file into a Document."""
    path = Path(path)

    if path.suffix != ".txt":
        raise ValueError(f"Only .txt files are supported, got '{path.suffix}'")

    content = path.read_text(encoding="utf-8")

    if not content.strip():
        raise ValueError(f"File is empty: {path}")

    stat = path.stat()
    metadata = {
        "filename": path.name,
        "file_path": str(path.resolve()),
        "file_size": stat.st_size,
        "char_count": len(content),
    }

    return Document(content=content, metadata=metadata)


def load_directory(path: str | Path) -> list[Document]:
    """Load all .txt files from a directory into a list of Documents."""
    path = Path(path)
    return [load_file(f) for f in sorted(path.glob("*.txt"))]
