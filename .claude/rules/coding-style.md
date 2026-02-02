# Coding Style Rules

일관된 코드 스타일을 유지하기 위한 규칙입니다.

## General Principles

### 1. Immutability First
```python
# ✅ Use frozen dataclasses for Value Objects
@dataclass(frozen=True)
class Document:
    content: str
    metadata: dict[str, str | int]

# ✅ Return new instances instead of mutating
def with_metadata(self, key: str, value: str) -> Document:
    return Document(self.content, {**self.metadata, key: value})
```

### 2. Single Responsibility
```python
# ❌ Too many responsibilities
class RAGManager:
    def load_documents(self): ...
    def split_chunks(self): ...
    def embed_and_store(self): ...
    def search_and_generate(self): ...

# ✅ Single responsibility
class VectorStore:
    def add_documents(self, docs: list[EmbeddedDocument]) -> list[str]: ...
    def search(self, query_embedding: list[float]) -> list[SearchResult]: ...

class Retriever:
    def retrieve(self, query: str) -> list[SearchResult]: ...
```

### 3. Explicit over Implicit
```python
# ❌ Implicit behavior
def process(data):
    return data.get('value', 0) * 100

# ✅ Explicit typing and handling
def process(data: dict[str, Any] | None) -> int:
    if data is None:
        raise ValueError("Data cannot be None")
    return int(data.get('value', 0)) * 100
```

## Python Conventions

### Naming
```python
# Classes: PascalCase
class VectorStore: ...

# Functions/methods: snake_case
def embed_documents(): ...

# Constants: UPPER_SNAKE_CASE
_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500

# Private: leading underscore
def _get_model(): ...
```

### Imports
```python
# Standard library
from dataclasses import dataclass
from pathlib import Path

# Third-party
from sentence_transformers import SentenceTransformer
import chromadb

# Local
from rag.loader import Document
from rag.embedder import embed_query
```

### Type Hints
```python
# ✅ Always use type hints
def load_directory(path: str | Path) -> list[Document]:
    ...

# ✅ Use | for Optional
def get_by_id(doc_id: str) -> Document | None:
    ...
```

## File Organization

### Max File Length
- **Hard limit**: 800 lines
- **Recommended**: 300-400 lines

### Max Function Length
- **Hard limit**: 50 lines
- **Recommended**: 20-30 lines

### Max Nesting Depth
- **Hard limit**: 4 levels
- Use early returns to reduce nesting

```python
# ❌ Deep nesting
def process(data):
    if data:
        if data.valid:
            if data.items:
                for item in data.items:
                    if item.active:
                        # process

# ✅ Early returns
def process(data):
    if not data or not data.valid:
        return None
    if not data.items:
        return []
    return [process_item(item) for item in data.items if item.active]
```
