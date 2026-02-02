"""TDD tests for Retriever (Step 5)."""

from pathlib import Path

from rag.embedder import EmbeddedDocument, embed_documents, embed_query
from rag.loader import Document, load_file
from rag.retriever import Retriever
from rag.splitter import split_document
from rag.store import SearchResult, VectorStore

DATA_DIR = Path(__file__).parent.parent / "data"


# --- Group A: 기본 Retrieve 동작 ---


class TestRetrieverBasic:
    def test_retrieve_returns_search_results(self):
        """Cycle 1: 문자열 쿼리 → SearchResult 리스트 반환."""
        store = VectorStore(collection_name="test_ret_basic")
        docs = [
            EmbeddedDocument(
                document=Document(
                    content="metformin treats type 2 diabetes",
                    metadata={"filename": "met.txt", "chunk_index": 0},
                ),
                embedding=embed_query("metformin treats type 2 diabetes"),
            ),
            EmbeddedDocument(
                document=Document(
                    content="aspirin is used for pain relief",
                    metadata={"filename": "asp.txt", "chunk_index": 0},
                ),
                embedding=embed_query("aspirin is used for pain relief"),
            ),
        ]
        store.add_documents(docs)

        retriever = Retriever(store=store, top_k=2)
        results = retriever.retrieve("diabetes medication")

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_top_k_limits_results(self):
        """Cycle 2: top_k=1 → 결과 1개만 반환."""
        store = VectorStore(collection_name="test_ret_topk")
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

        retriever = Retriever(store=store, top_k=3)
        results = retriever.retrieve("document 0", top_k=1)

        assert len(results) == 1

    def test_semantic_relevance(self):
        """Cycle 3: 의미적으로 관련된 문서가 상위에 위치."""
        store = VectorStore(collection_name="test_ret_semantic")
        docs = [
            EmbeddedDocument(
                document=Document(
                    content="metformin is a medication for type 2 diabetes",
                    metadata={"filename": "met.txt", "chunk_index": 0},
                ),
                embedding=embed_query(
                    "metformin is a medication for type 2 diabetes"
                ),
            ),
            EmbeddedDocument(
                document=Document(
                    content="the weather forecast shows rain tomorrow",
                    metadata={"filename": "weather.txt", "chunk_index": 0},
                ),
                embedding=embed_query("the weather forecast shows rain tomorrow"),
            ),
        ]
        store.add_documents(docs)

        retriever = Retriever(store=store, top_k=2)
        results = retriever.retrieve("diabetes treatment")

        assert results[0].document.metadata["filename"] == "met.txt"


# --- Group B: Score Threshold 필터링 ---


class TestScoreThreshold:
    def test_threshold_filters_low_scores(self):
        """Cycle 4: score_threshold 이상인 결과만 반환."""
        store = VectorStore(collection_name="test_ret_threshold")
        docs = [
            EmbeddedDocument(
                document=Document(
                    content="metformin is used for diabetes management",
                    metadata={"filename": "met.txt", "chunk_index": 0},
                ),
                embedding=embed_query("metformin is used for diabetes management"),
            ),
            EmbeddedDocument(
                document=Document(
                    content="the sun rises in the east every morning",
                    metadata={"filename": "sun.txt", "chunk_index": 0},
                ),
                embedding=embed_query("the sun rises in the east every morning"),
            ),
        ]
        store.add_documents(docs)

        retriever = Retriever(store=store, top_k=2, score_threshold=0.3)
        results = retriever.retrieve("diabetes medication")

        assert all(r.score >= 0.3 for r in results)

    def test_threshold_zero_returns_all(self):
        """Cycle 5: threshold=0.0(기본값)이면 필터 없이 전부 반환."""
        store = VectorStore(collection_name="test_ret_no_threshold")
        docs = [
            EmbeddedDocument(
                document=Document(
                    content="random text about cooking recipes",
                    metadata={"filename": "cook.txt", "chunk_index": 0},
                ),
                embedding=embed_query("random text about cooking recipes"),
            ),
        ]
        store.add_documents(docs)

        retriever = Retriever(store=store, top_k=5)
        results = retriever.retrieve("quantum physics")

        assert len(results) == 1  # 관련 없지만 threshold=0이므로 반환됨


# --- Group C: 에지 케이스 ---


class TestRetrieverEdgeCases:
    def test_retrieve_empty_store(self):
        """Cycle 6: 빈 store에서 retrieve → 빈 리스트."""
        store = VectorStore(collection_name="test_ret_empty")
        retriever = Retriever(store=store)
        results = retriever.retrieve("anything")
        assert results == []

    def test_override_top_k_at_call(self):
        """Cycle 7: retrieve() 호출 시 top_k 오버라이드."""
        store = VectorStore(collection_name="test_ret_override")
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

        retriever = Retriever(store=store, top_k=5)
        results = retriever.retrieve("document", top_k=2)
        assert len(results) == 2

    def test_override_threshold_at_call(self):
        """Cycle 8: retrieve() 호출 시 score_threshold 오버라이드."""
        store = VectorStore(collection_name="test_ret_override_thr")
        docs = [
            EmbeddedDocument(
                document=Document(
                    content="metformin treats diabetes",
                    metadata={"filename": "met.txt", "chunk_index": 0},
                ),
                embedding=embed_query("metformin treats diabetes"),
            ),
            EmbeddedDocument(
                document=Document(
                    content="the moon orbits the earth",
                    metadata={"filename": "moon.txt", "chunk_index": 0},
                ),
                embedding=embed_query("the moon orbits the earth"),
            ),
        ]
        store.add_documents(docs)

        retriever = Retriever(store=store, top_k=2, score_threshold=0.0)
        results = retriever.retrieve("diabetes", score_threshold=0.3)
        assert all(r.score >= 0.3 for r in results)


# --- Group D: 통합 ---


class TestRetrieverIntegration:
    def test_full_pipeline_with_retriever(self):
        """Cycle 9: load → split → embed → store → retrieve 전체 흐름."""
        doc = load_file(DATA_DIR / "metformin_overview.txt")
        chunks = split_document(doc, chunk_size=300, chunk_overlap=30)
        embedded = embed_documents(chunks)

        store = VectorStore(collection_name="test_ret_pipeline")
        store.add_documents(embedded)

        retriever = Retriever(store=store, top_k=3, score_threshold=0.1)
        results = retriever.retrieve("what are the side effects of metformin?")

        assert len(results) > 0
        assert len(results) <= 3
        assert all(r.score >= 0.1 for r in results)
        assert results[0].score >= results[-1].score
        assert results[0].document.metadata["filename"] == "metformin_overview.txt"
