"""TDD tests for Document Loading (Step 1)."""

from pathlib import Path

import pytest

from rag.loader import Document, load_directory, load_file

DATA_DIR = Path(__file__).parent.parent / "data"


# --- Group A: Document 모델 ---


class TestDocument:
    def test_document_creation(self):
        """Cycle 1: content + metadata로 Document 생성."""
        doc = Document(content="hello", metadata={"source": "test"})
        assert doc.content == "hello"
        assert doc.metadata == {"source": "test"}

    def test_document_default_metadata(self):
        """metadata 미지정 시 빈 dict."""
        doc = Document(content="hello")
        assert doc.metadata == {}


# --- Group B: 단일 파일 로딩 ---


class TestLoadFile:
    def test_load_file_returns_document(self):
        """Cycle 2: load_file은 Document 인스턴스를 반환."""
        result = load_file(DATA_DIR / "metformin_overview.txt")
        assert isinstance(result, Document)

    def test_load_file_content_matches(self):
        """Cycle 3: 파일 내용이 Document.content와 일치."""
        path = DATA_DIR / "metformin_overview.txt"
        expected = path.read_text(encoding="utf-8")
        doc = load_file(path)
        assert doc.content == expected

    def test_load_file_metadata(self):
        """Cycle 4: metadata에 filename, file_path, file_size, char_count 포함."""
        path = DATA_DIR / "metformin_overview.txt"
        doc = load_file(path)

        assert doc.metadata["filename"] == "metformin_overview.txt"
        assert doc.metadata["file_path"] == str(path.resolve())
        assert isinstance(doc.metadata["file_size"], int)
        assert doc.metadata["file_size"] > 0
        assert doc.metadata["char_count"] == len(doc.content)


# --- Group C: 디렉토리 로딩 ---


class TestLoadDirectory:
    def test_load_directory_returns_list(self):
        """Cycle 5: load_directory는 Document 리스트를 반환."""
        result = load_directory(DATA_DIR)
        assert isinstance(result, list)
        assert all(isinstance(doc, Document) for doc in result)

    def test_load_directory_count(self):
        """Cycle 6: data/ 디렉토리의 .txt 파일 3개를 모두 로드."""
        result = load_directory(DATA_DIR)
        assert len(result) == 3

    def test_load_directory_only_txt_files(self, tmp_path: Path):
        """Cycle 7: .txt 파일만 로드, 다른 확장자는 무시."""
        (tmp_path / "a.txt").write_text("txt file", encoding="utf-8")
        (tmp_path / "b.pdf").write_text("pdf file", encoding="utf-8")
        (tmp_path / "c.md").write_text("md file", encoding="utf-8")

        result = load_directory(tmp_path)
        assert len(result) == 1
        assert result[0].metadata["filename"] == "a.txt"


# --- Group D: 에러 처리 ---


class TestLoadFileErrors:
    def test_load_file_not_found(self):
        """Cycle 8: 존재하지 않는 파일 → FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_file(Path("/nonexistent/file.txt"))

    def test_load_file_empty(self, tmp_path: Path):
        """Cycle 9: 빈 파일 → ValueError('empty')."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="empty"):
            load_file(empty_file)

    def test_load_file_not_txt(self, tmp_path: Path):
        """Cycle 10: .txt가 아닌 파일 → ValueError('.txt')."""
        non_txt = tmp_path / "data.csv"
        non_txt.write_text("some,data", encoding="utf-8")

        with pytest.raises(ValueError, match=r"\.txt"):
            load_file(non_txt)
