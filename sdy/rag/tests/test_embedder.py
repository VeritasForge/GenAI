"""TDD tests for Embedding (Step 3)."""

import numpy as np
import pytest

from rag.embedder import EmbeddedDocument, embed_documents, embed_query
from rag.loader import Document
from rag.splitter import split_document

# --- Group A: 기본 기능 ---


class TestEmbedBasic:
    def test_embedded_document_creation(self):
        """Cycle 1: EmbeddedDocument는 document + embedding을 보유."""
        doc = Document(content="hello world")
        ed = EmbeddedDocument(document=doc, embedding=[0.1, 0.2, 0.3])
        assert ed.document is doc
        assert ed.embedding == [0.1, 0.2, 0.3]

    def test_embed_query_returns_vector(self):
        """Cycle 2: embed_query는 float 리스트를 반환."""
        vector = embed_query("what is metformin?")
        assert isinstance(vector, list)
        assert all(isinstance(v, float) for v in vector)

    def test_embed_query_dimension(self):
        """Cycle 3: 벡터 차원은 384."""
        vector = embed_query("what is aspirin?")
        assert len(vector) == 384

    def test_embed_documents_returns_embedded_list(self):
        """Cycle 4: embed_documents는 list[EmbeddedDocument]를 반환."""
        docs = [
            Document(content="metformin is a drug"),
            Document(content="aspirin reduces pain"),
        ]
        result = embed_documents(docs)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(ed, EmbeddedDocument) for ed in result)
        assert all(len(ed.embedding) == 384 for ed in result)


# --- Group B: 순서 & Metadata & 유사도 ---


class TestOrderMetadataSimilarity:
    def test_order_preservation(self):
        """Cycle 5: embedded[i].document == docs[i] 순서 보존."""
        docs = [
            Document(content="first document"),
            Document(content="second document"),
            Document(content="third document"),
        ]
        result = embed_documents(docs)
        for i, ed in enumerate(result):
            assert ed.document is docs[i]

    def test_metadata_preserved(self):
        """Cycle 6: 원본 metadata가 그대로 보존."""
        docs = [
            Document(
                content="metformin overview",
                metadata={"filename": "metformin.txt", "chunk_index": 0},
            ),
        ]
        result = embed_documents(docs)
        assert result[0].document.metadata["filename"] == "metformin.txt"
        assert result[0].document.metadata["chunk_index"] == 0

    def test_similar_texts_closer(self):
        """Cycle 7: 의미가 비슷한 텍스트는 cosine 유사도가 높다."""
        v_drug = np.array(embed_query("metformin is used for diabetes treatment"))
        v_similar = np.array(embed_query("insulin helps control blood sugar"))
        v_unrelated = np.array(embed_query("the weather is sunny today"))

        def cosine_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        sim_related = cosine_sim(v_drug, v_similar)
        sim_unrelated = cosine_sim(v_drug, v_unrelated)
        assert sim_related > sim_unrelated


# --- Group C: 에러 처리 ---


class TestEmbedErrors:
    def test_empty_list_returns_empty(self):
        """Cycle 8: 빈 리스트 입력 → 빈 리스트 반환."""
        result = embed_documents([])
        assert result == []

    def test_empty_content_raises(self):
        """Cycle 9: 빈 content → ValueError."""
        docs = [Document(content="")]
        with pytest.raises(ValueError, match="empty"):
            embed_documents(docs)


# --- Group D: 통합 ---


class TestIntegration:
    def test_split_then_embed(self):
        """Cycle 10: splitter → embedder 파이프라인 연결."""
        doc = Document(
            content="Metformin is a medication used to treat type 2 diabetes. " * 20,
            metadata={"filename": "metformin.txt"},
        )
        chunks = split_document(doc, chunk_size=200, chunk_overlap=30)
        assert len(chunks) >= 2

        embedded = embed_documents(chunks)
        assert len(embedded) == len(chunks)
        for ed in embedded:
            assert len(ed.embedding) == 384
            assert ed.document.metadata["filename"] == "metformin.txt"
            assert "chunk_index" in ed.document.metadata
