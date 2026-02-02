"""TDD tests for Text Splitting (Step 2)."""

import pytest

from rag.loader import Document
from rag.splitter import split_document, split_documents

# --- Group A: 기본 분할 ---


class TestSplitBasic:
    def test_split_returns_list_of_documents(self):
        """Cycle 1: split_document은 list[Document]를 반환."""
        doc = Document(content="hello world")
        result = split_document(doc)
        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)

    def test_short_text_single_chunk(self):
        """Cycle 2: chunk_size 이하 텍스트 → chunk 1개."""
        doc = Document(content="short text")
        result = split_document(doc, chunk_size=500)
        assert len(result) == 1
        assert result[0].content == "short text"

    def test_long_text_multiple_chunks(self):
        """Cycle 3: chunk_size보다 긴 텍스트 → 여러 chunk."""
        doc = Document(content="a" * 1000)
        result = split_document(doc, chunk_size=500, chunk_overlap=0)
        assert len(result) == 2

    def test_chunk_size_respected(self):
        """Cycle 4: 모든 chunk는 chunk_size 이하."""
        doc = Document(content="a" * 2000)
        result = split_document(doc, chunk_size=500, chunk_overlap=50)
        for chunk in result:
            assert len(chunk.content) <= 500


# --- Group B: Overlap & Metadata ---


class TestOverlapAndMetadata:
    def test_overlap_content(self):
        """Cycle 5: chunk[1]의 시작이 chunk[0]의 끝 50자와 동일."""
        doc = Document(content="a" * 300 + "b" * 300)
        result = split_document(doc, chunk_size=500, chunk_overlap=50)
        assert len(result) >= 2
        # chunk[0]의 마지막 50자 == chunk[1]의 첫 50자
        assert result[0].content[-50:] == result[1].content[:50]

    def test_metadata_propagation(self):
        """Cycle 6: 원본 metadata 상속 + chunk 고유 정보."""
        doc = Document(
            content="a" * 1000,
            metadata={"filename": "test.txt", "file_size": 1000},
        )
        result = split_document(doc, chunk_size=500, chunk_overlap=50)

        for i, chunk in enumerate(result):
            # 원본 metadata 상속
            assert chunk.metadata["filename"] == "test.txt"
            assert chunk.metadata["file_size"] == 1000
            # chunk 고유 정보
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["chunk_count"] == len(result)
            assert chunk.metadata["chunk_char_count"] == len(chunk.content)


# --- Group C: 일괄 처리 ---


class TestSplitDocuments:
    def test_split_documents_multiple(self):
        """Cycle 7: 여러 Document를 일괄 분할."""
        docs = [
            Document(content="a" * 1000, metadata={"filename": "a.txt"}),
            Document(content="b" * 800, metadata={"filename": "b.txt"}),
            Document(content="c" * 100, metadata={"filename": "c.txt"}),
        ]
        result = split_documents(docs, chunk_size=500, chunk_overlap=0)

        # a: 2 chunks, b: 2 chunks, c: 1 chunk = 5 total
        assert len(result) == 5
        # 각 chunk가 올바른 원본 파일을 추적
        filenames = [d.metadata["filename"] for d in result]
        assert filenames.count("a.txt") == 2
        assert filenames.count("b.txt") == 2
        assert filenames.count("c.txt") == 1


# --- Group D: 에러 처리 ---


class TestSplitErrors:
    def test_empty_content_raises(self):
        """Cycle 8: 빈 content → ValueError."""
        doc = Document(content="")
        with pytest.raises(ValueError, match="empty"):
            split_document(doc)

    def test_invalid_chunk_size(self):
        """Cycle 9: chunk_size <= 0 → ValueError."""
        doc = Document(content="hello")
        with pytest.raises(ValueError, match="chunk_size"):
            split_document(doc, chunk_size=0)

    def test_overlap_exceeds_chunk_size(self):
        """Cycle 10: overlap >= chunk_size → ValueError."""
        doc = Document(content="hello world")
        with pytest.raises(ValueError, match="overlap"):
            split_document(doc, chunk_size=10, chunk_overlap=10)
