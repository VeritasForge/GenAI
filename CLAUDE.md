# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 1. Repository Structure

Gen AI 학습 프로젝트 모노레포. 현재 포함:

- `sdy/rag/` — 순수 Python으로 구현한 RAG (Retrieval-Augmented Generation) 파이프라인. LangChain/LlamaIndex 없이 ChromaDB + sentence-transformers + Claude Code CLI를 LLM 백엔드로 사용.

## 2. Commands (sdy/rag)

모든 명령은 `sdy/rag/` 디렉토리에서 실행:

```bash
# 의존성 설치
uv sync --group dev

# 테스트 실행
uv run pytest
uv run pytest tests/test_loader.py                                    # 단일 파일
uv run pytest tests/test_loader.py -k "test_load_file_returns_document"  # 단일 테스트

# 린트 & 포맷
uv run ruff check src tests --fix
uv run ruff format src tests

# CLI
uv run rag hello          # smoke test
uv run rag index           # 문서 인덱싱 (./data)
uv run rag ask "question"  # 단일 질의
uv run rag chat            # 대화형 REPL
```

## 3. Architecture (sdy/rag)

파이프라인은 선형 데이터 흐름을 따르며, `src/rag/`의 각 모듈이 하나의 단계를 담당:

```
loader.py → splitter.py → embedder.py → store.py → retriever.py → generator.py
 (load)      (chunk)       (embed)      (ChromaDB)   (search)      (LLM call)
```

- **`loader.py`** — `Document` dataclass (content + metadata), `.txt` 파일 로드
- **`splitter.py`** — 문자 단위 청킹, 크기/오버랩 설정 가능
- **`embedder.py`** — Singleton `SentenceTransformer` (`all-MiniLM-L6-v2`, 384-dim), lazy-load
- **`store.py`** — `VectorStore` (ChromaDB, cosine similarity); `SearchResult` dataclass
- **`retriever.py`** — `Retriever` facade: query string → embed → vector search
- **`generator.py`** — `Generator`: `claude -p` subprocess 호출로 LLM 답변 생성
- **`pipeline.py`** — 오케스트레이션: `index_documents()` (load→store), `ask_question()` (retrieve→generate)
- **`cli.py`** — Typer CLI: `index`, `ask`, `chat`, `hello` 명령

핵심 설계: `pipeline.py`는 CLI 의존 없는 순수 로직 레이어. `cli.py`는 파이프라인 함수를 호출하고 출력을 포맷하는 얇은 래퍼.

## 4. Conventions

- **Python 3.12+**, src layout (`src/rag/`), `uv` 패키지 매니저
- **Linting**: ruff (rules: `E, F, I, W`; line-length: 88; double quotes)
- **Pre-commit**: trailing whitespace, end-of-file fixer, ruff check+format
- **Build**: hatchling backend
- **Environment**: `.env` 파일에 `OPENAI_API_KEY` (`.env.example` 참조). 현재 generation은 Claude Code CLI 사용

---

## 5. Core Principles

### 5.1 TDD Protocol
Robert C. Martin의 TDD 3법칙을 따른다:
1. **Red**: 실패하는 단위 테스트 없이 프로덕션 코드를 작성하지 않는다.
2. **Green**: 테스트를 통과하는 최소한의 코드만 작성한다.
3. **Refactor**: 리팩토링 중에 기능을 추가하지 않는다.

테스트 구조: **Given-When-Then** 패턴, 명시적 주석.
테스트 명명: `test_should_[expected_behavior]_when_[condition]`

### 5.2 Modern Python
- **Type Hints**: 내장 제네릭 (`list[str]`, `dict`, `str | None`) 사용. `typing` 모듈 별칭 지양.
- **Naming**: snake_case (변수/함수), PascalCase (클래스), UPPER_SNAKE_CASE (상수)

### 5.3 Clean Code
- **의도를 드러내는 이름**: `days_since_creation` > `d`
- **작은 함수**: 하나의 일만 수행 (50줄 이하)
- **파일 크기**: 800줄 이하 권장
- **중첩 깊이**: 4단계 이하 (early return 활용)
- **부작용 없음**: 숨겨진 상태 변경 금지
- **최소 주석**: 코드가 스스로 설명, 주석은 "Why"만

### 5.4 코딩 스타일
- **불변성 우선**: frozen dataclass, 새 인스턴스 반환
- **단일 책임 원칙 (SRP)**: 클래스/함수는 변경의 이유가 하나
- **명시적 > 암묵적**: 타입 힌트, 명시적 에러 핸들링

### 5.5 Documentation Language (Korean)
모든 문서는 **한국어**로 작성. 기술 용어(API, TDD, RAG 등)와 약어는 영어 유지 가능.

### 5.6 AI Interaction Protocols
1. **Edge Cases & Error Handling**: 항상 경계 케이스와 에러 처리를 고려.
2. **Deep Thinking**: 복잡한 분석 시 `sequentialthinking` MCP 도구 사용.
3. **Interactive Clarification**: 불확실한 부분은 가정하지 말고 사용자에게 확인.

## 6. AI 사고 프로세스 (Chain of Thought)

복잡한 문제 해결이나 설계 결정이 필요한 경우, 반드시 `sequentialthinking` MCP 도구를 사용하여 다음 단계를 거친다:

1. **상황 분석**: 현재 요청과 관련된 컨텍스트, 제약 조건, 관련 파일 파악
2. **전략 수립**: 가능한 해결책 나열, 장단점 비교, 최적 전략 선택
3. **단계별 계획**: 선택한 전략의 구체적 실행 단계 정의
4. **검증 및 회고**: 계획이 요구사항을 충족하는지, 누락은 없는지 검토

## 7. Test Protection Protocol

코드 변경 시 기존 테스트 보호를 위한 **필수** 워크플로우.

### 변경 전 (Pre-Change)
1. 기존 테스트 실행으로 베이스라인 확보: `uv run pytest -v` (sdy/rag/)
2. 모든 테스트 PASS 확인. FAIL이 있으면 먼저 수정 후 진행.

### 변경 후 (Post-Change)
1. **전체 테스트 스위트** 재실행 (변경 범위 무관)
2. 변경 전 PASS → 변경 후 FAIL: **Breaking Change 감지** → 작업 중단, 사용자에게 보고
3. **자동 수정 금지**: 사용자 승인 없이 실패한 테스트를 수정/삭제/주석 처리하지 않음

## 8. Claude Code Configuration

### 8.1 Directory Structure
```
.claude/
├── settings.local.json    # 권한, hooks, 모델 설정
├── agents/                # 실행 에이전트
│   ├── tdd-developer.md   # RED → GREEN → REFACTOR
│   ├── code-reviewer.md   # 코드 품질 검토
│   ├── test-reviewer.md   # 테스트 품질 검토
│   └── security-reviewer.md # 보안 검토
├── commands/              # 슬래시 명령어
│   ├── tdd.md             # /tdd - 빠른 TDD
│   ├── review.md          # /review - 코드 리뷰
│   ├── build-fix.md       # /build-fix
│   ├── wrap.md            # /wrap - 문서 업데이트
│   └── commit.md          # /commit - 커밋 및 푸시
└── rules/                 # 항상 준수할 규칙
    ├── security.md        # 보안 규칙
    ├── coding-style.md    # 코딩 스타일
    ├── testing.md         # 테스트 규칙
    └── git-workflow.md    # Git 워크플로우
```

### 8.2 Development Agents

| Agent | 역할 | 실행 방식 |
|-------|------|----------|
| `tdd-developer` | RED → GREEN → REFACTOR 수행 | 순차 (작업별) |
| `code-reviewer` | 코드 품질 검토 | 병렬 (리뷰 시) |
| `test-reviewer` | 테스트 품질 검토 | 병렬 (리뷰 시) |
| `security-reviewer` | 보안 검토 | 병렬 (리뷰 시) |

> **중요**: Subagent는 **zero context**로 시작. prompt에 필요한 정보를 명시적으로 전달해야 함.

### 8.3 Available Commands

| Command | Description |
|---------|-------------|
| `/tdd` | 빠른 TDD 워크플로우 |
| `/review` | 코드 리뷰 실행 |
| `/build-fix` | 빌드 오류 진단 및 수정 |
| `/wrap` | 프로젝트 문서 업데이트 |
| `/commit` | 변경사항 커밋 및 푸시 |

### 8.4 Key Rules (Always Follow)
1. **Security**: 하드코딩된 비밀 금지, 입력 검증 필수
2. **Testing**: TDD 준수, Given-When-Then 구조
3. **Coding Style**: 불변성 우선, 단일 책임 원칙, 명시적 타이핑
4. **Git**: Conventional Commits, 작은 단위 커밋, 테스트와 구현 함께 커밋

자세한 규칙은 `.claude/rules/` 내 파일 참조.
