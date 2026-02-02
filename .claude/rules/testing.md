# Testing Rules

테스트 작성에 관한 필수 규칙입니다.

## Core Principle

> **No code without tests. Tests are not optional.**

## TDD Workflow

```
1. RED    → 실패하는 테스트 작성
2. GREEN  → 테스트 통과하는 최소 코드
3. REFACTOR → 코드 개선 (테스트 유지)
```

## Test Structure

### Given-When-Then Pattern
```python
def test_should_split_document_into_chunks():
    # Given - 테스트 데이터 준비
    doc = Document(content="A" * 1000, metadata={"filename": "test.txt"})

    # When - 테스트 대상 실행
    chunks = split_document(doc, chunk_size=500, chunk_overlap=50)

    # Then - 결과 검증
    assert len(chunks) == 3
    assert all(len(c.content) <= 500 for c in chunks)
```

### Descriptive Test Names
```python
# ❌ Vague names
def test_load():
def test_split():

# ✅ Descriptive names
def test_should_load_txt_file_with_metadata():
def test_should_raise_when_chunk_size_is_zero():
def test_should_return_empty_list_when_no_documents():
```

## Edge Cases Checklist

모든 함수에 대해 다음 케이스 테스트:

- [ ] **None/empty** 입력
- [ ] **빈 리스트/딕셔너리**
- [ ] **잘못된 타입**
- [ ] **경계값** (0, 음수, 최대값)

## Anti-Patterns to Avoid

### 1. Testing Implementation Details
```python
# ❌ Tests private method
def test_internal():
    store = VectorStore()
    result = store._make_id(...)  # Private!

# ✅ Tests public interface
def test_add_and_search():
    store = VectorStore()
    store.add_documents([doc])
    results = store.search(query_embedding)
    assert len(results) > 0
```

### 2. Test Dependencies
```python
# ❌ Tests depend on execution order
def test_add():
    store.add_documents([doc])

def test_search():
    results = store.search(query)  # Depends on test above

# ✅ Independent tests
def test_add_and_search():
    store = VectorStore()
    store.add_documents([doc])
    results = store.search(query)
    assert len(results) == 1
```

### 3. Over-Mocking
```python
# ❌ Mocks everything
def test_pipeline():
    loader = Mock()
    splitter = Mock()
    embedder = Mock()
    # What are we even testing?

# ✅ Mock only external dependencies
def test_retriever_returns_relevant_docs():
    store = Mock()
    store.search.return_value = [SearchResult(...)]
    retriever = Retriever(store=store)
    results = retriever.retrieve("query")
    assert len(results) == 1
```

## Test Commands

```bash
# sdy/rag/ 디렉토리에서 실행
uv run pytest -v
uv run pytest tests/test_loader.py -v
uv run pytest -k "test_should_load" -v
```
