"""TDD tests for Vector Store (Step 4)."""

from pathlib import Path

from rag.embedder import EmbeddedDocument, embed_documents, embed_query
from rag.loader import Document, load_file
from rag.splitter import split_document
from rag.store import SearchResult, VectorStore

DATA_DIR = Path(__file__).parent.parent / "data"


# --- Group A: 기본 Store 동작 ---


class TestStoreBasic:
    def test_search_result_creation(self):
        """Cycle 1: SearchResult는 document + score를 보유."""
        doc = Document(content="hello world")
        sr = SearchResult(document=doc, score=0.95)
        assert sr.document is doc
        assert sr.score == 0.95

    def test_store_creation_empty(self):
        """Cycle 2: 새 store는 count() == 0."""
        store = VectorStore(collection_name="test_empty")
        assert store.count() == 0

    def test_add_documents_increases_count(self):
        """Cycle 3: 2개 추가 → count==2, ID 리스트 반환."""
        store = VectorStore(collection_name="test_add")
        docs = [
            EmbeddedDocument(
                document=Document(
                    content="metformin overview",
                    metadata={"filename": "met.txt", "chunk_index": 0},
                ),
                embedding=[0.1] * 384,
            ),
            EmbeddedDocument(
                document=Document(
                    content="aspirin overview",
                    metadata={"filename": "asp.txt", "chunk_index": 0},
                ),
                embedding=[0.2] * 384,
            ),
        ]
        ids = store.add_documents(docs)
        assert len(ids) == 2
        assert store.count() == 2

    def test_search_returns_results(self):
        """Cycle 4: 5개 저장 후 top_k=3 검색 → 3개 반환."""
        store = VectorStore(collection_name="test_search")
        docs = [
            EmbeddedDocument(
                document=Document(
                    content=f"document {i}",
                    metadata={"filename": "test.txt", "chunk_index": i},
                ),
                embedding=embed_query(f"document {i}"),
            )
            for i in range(5)
        ]
        store.add_documents(docs)

        query_vec = embed_query("document 0")
        results = store.search(query_vec, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(isinstance(r.score, float) for r in results)


# --- Group B: 검색 품질 & Metadata ---


class TestSearchQualityAndMetadata:
    def test_semantic_ordering(self):
        """Cycle 5: 'diabetes' 쿼리에 metformin chunk가 weather보다 상위."""
        store = VectorStore(collection_name="test_semantic")
        docs = [
            EmbeddedDocument(
                document=Document(
                    content="metformin is a medication used to treat type 2 diabetes",
                    metadata={"filename": "met.txt", "chunk_index": 0},
                ),
                embedding=embed_query(
                    "metformin is a medication used to treat type 2 diabetes"
                ),
            ),
            EmbeddedDocument(
                document=Document(
                    content="the weather today is sunny and warm",
                    metadata={"filename": "weather.txt", "chunk_index": 0},
                ),
                embedding=embed_query("the weather today is sunny and warm"),
            ),
            EmbeddedDocument(
                document=Document(
                    content="aspirin is used for pain relief and fever",
                    metadata={"filename": "asp.txt", "chunk_index": 0},
                ),
                embedding=embed_query("aspirin is used for pain relief and fever"),
            ),
        ]
        store.add_documents(docs)

        query_vec = embed_query("diabetes treatment medication")
        results = store.search(query_vec, top_k=3)

        assert results[0].document.metadata["filename"] == "met.txt"
        assert results[0].score > results[-1].score

    def test_metadata_preserved(self):
        """Cycle 6: filename, chunk_index가 저장-검색 왕복 후에도 보존."""
        store = VectorStore(collection_name="test_metadata")
        docs = [
            EmbeddedDocument(
                document=Document(
                    content="metformin overview text",
                    metadata={"filename": "metformin.txt", "chunk_index": 2},
                ),
                embedding=embed_query("metformin overview text"),
            ),
        ]
        store.add_documents(docs)

        query_vec = embed_query("metformin")
        results = store.search(query_vec, top_k=1)

        assert results[0].document.metadata["filename"] == "metformin.txt"
        assert results[0].document.metadata["chunk_index"] == 2

    def test_deterministic_ids(self):
        """Cycle 7: metadata 기반 결정적 ID 생성 'test.txt::chunk-3'."""
        store = VectorStore(collection_name="test_ids")
        doc = EmbeddedDocument(
            document=Document(
                content="some content",
                metadata={"filename": "test.txt", "chunk_index": 3},
            ),
            embedding=[0.1] * 384,
        )
        ids = store.add_documents([doc])
        assert ids[0] == "test.txt::chunk-3"


# --- Group C: 에러 처리 ---


class TestStoreEdgeCases:
    def test_search_empty_store(self):
        """Cycle 8: 빈 store 검색 → 빈 리스트."""
        store = VectorStore(collection_name="test_empty_search")
        query_vec = embed_query("anything")
        results = store.search(query_vec, top_k=5)
        assert results == []

    def test_add_empty_list(self):
        """Cycle 9: 빈 리스트 추가 → 빈 ID 리스트, count 불변."""
        store = VectorStore(collection_name="test_add_empty")
        ids = store.add_documents([])
        assert ids == []
        assert store.count() == 0


# --- Group D: 통합 ---


class TestIntegration:
    def test_full_pipeline(self):
        """Cycle 10: load → split → embed → store → search 전체 파이프라인."""
        doc = load_file(DATA_DIR / "metformin_overview.txt")
        chunks = split_document(doc, chunk_size=300, chunk_overlap=30)
        embedded = embed_documents(chunks)

        store = VectorStore(collection_name="test_pipeline")
        ids = store.add_documents(embedded)

        assert len(ids) == len(embedded)
        assert store.count() == len(embedded)

        query_vec = embed_query("what are the side effects of metformin?")
        results = store.search(query_vec, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.score > 0 for r in results)
        assert results[0].score >= results[1].score >= results[2].score
        assert results[0].document.metadata["filename"] == "metformin_overview.txt"
