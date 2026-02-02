# /tdd - Test-Driven Development Workflow

TDD 방법론에 따라 기능을 구현합니다.

## Usage

```
/tdd [기능 설명]
```

## Workflow

1. **인터페이스 정의**: 먼저 타입/인터페이스를 정의합니다
2. **RED**: 실패하는 테스트를 작성합니다
3. **GREEN**: 테스트를 통과하는 최소한의 코드를 작성합니다
4. **REFACTOR**: 코드를 개선합니다 (테스트는 계속 통과해야 함)

## Example

```
/tdd PDF 파일 로딩 기능 추가
```

## Steps

### Step 1: Write Failing Test (RED)
```python
def test_should_load_pdf_file_with_metadata():
    # Given
    pdf_path = Path("tests/fixtures/sample.pdf")

    # When
    doc = load_file(pdf_path)

    # Then
    assert doc.content != ""
    assert doc.metadata["filename"] == "sample.pdf"
```

### Step 2: Implement (GREEN)
```python
# 최소한의 구현
def load_file(path: str | Path) -> Document:
    path = Path(path)
    if path.suffix == ".pdf":
        return _load_pdf(path)
    # ...existing .txt logic...
```

### Step 3: Refactor
```python
# 코드 개선 (테스트 유지)
```

## Test Commands

```bash
# sdy/rag/ 디렉토리에서
uv run pytest tests/ -v
uv run pytest tests/test_loader.py -v
uv run pytest tests/ -k "test_should_load" -v
```
