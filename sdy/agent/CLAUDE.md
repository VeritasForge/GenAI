# CLAUDE.md

이 파일은 Claude Code가 `sdy/agent/` 프로젝트 작업 시 참조하는 가이드입니다.

---

## 1. 프로젝트 개요

PubMed와 ClinicalTrials.gov를 활용하는 멀티스텝 ReAct 에이전트.
Claude Code CLI를 LLM 백엔드로 사용하며, Tool을 자율적으로 호출하여 약물 연구 질문에 답한다.

## 2. 명령어

모든 명령은 `sdy/agent/` 디렉토리에서 실행:

```bash
# 의존성 설치
uv sync --group dev

# 테스트 실행
uv run pytest
uv run pytest tests/test_agent.py                          # 단일 파일
uv run pytest tests/test_agent.py -k "test_should_fallback"  # 단일 테스트

# 린트 & 포맷
uv run ruff check src tests --fix
uv run ruff format src tests

# CLI
uv run agent hello              # smoke test
uv run agent ask "질문"          # 단일 질의
uv run agent research "약물명"   # 종합 리포트
```

## 3. 아키텍처

ReAct (Reasoning + Acting) 패턴 기반 에이전트.

```
cli.py → pipeline.py → Agent (core/agent.py)
                          │
                          ├─→ call_llm() (llm/client.py) → Claude CLI
                          ├─→ parse_llm_response() (core/parser.py)
                          └─→ ToolRegistry (tools/registry.py)
                                ├── search_pubmed (tools/pubmed.py)
                                ├── fetch_article_details (tools/pubmed.py)
                                └── search_trials (tools/clinical_trials.py)
```

### 모듈별 역할

| 모듈 | 역할 |
|------|------|
| `cli.py` | Typer CLI: `ask`, `research`, `hello` 명령 |
| `pipeline.py` | 오케스트레이션: Tool 등록 + Agent 실행 |
| `report.py` | 수집 데이터를 구조화된 리포트로 변환 |
| `core/agent.py` | ReAct Loop: LLM 호출 → 파싱 → Tool 실행 반복 |
| `core/parser.py` | LLM 응답 JSON 파서 (AgentAction/AgentFinish 추출) |
| `core/types.py` | 불변 dataclass: AgentAction, AgentFinish, AgentStep, AgentResult |
| `llm/client.py` | Claude CLI subprocess 래퍼 (`claude -p`) |
| `llm/prompts.py` | 시스템 프롬프트, JSON 포맷 지시 |
| `tools/registry.py` | Tool 등록/실행/프롬프트 생성 |
| `tools/pubmed.py` | PubMed API (NCBI E-utilities) |
| `tools/clinical_trials.py` | ClinicalTrials.gov v2 API |
| `tools/types.py` | 불변 dataclass: Article, Trial |

### 핵심 설계 결정

- **LLM 백엔드**: Claude Code CLI (`claude -p --append-system-prompt`)
  - Claude Code CLI는 내부 시스템 프롬프트를 항상 유지하므로, JSON 응답 강제가 불가능
  - LLM이 자연어로 응답하면 fallback으로 `AgentFinish(answer=raw_response)` 처리
- **Tool Registry**: 런타임에 Tool을 동적으로 등록/해제 가능
- **파이프라인 분리**: `pipeline.py`는 CLI 의존 없는 순수 로직. `cli.py`는 얇은 래퍼

## 4. 규약

- **Python 3.12+**, src layout (`src/agent/`), `uv` 패키지 매니저
- **Linting**: ruff (rules: `E, F, I, W`; line-length: 88; double quotes)
- **Build**: hatchling backend
- **외부 API**: PubMed (NCBI E-utilities), ClinicalTrials.gov v2 — 별도 API 키 불요

## 5. 핵심 원칙

### 5.1 TDD Protocol
Robert C. Martin의 TDD 3법칙:
1. **Red**: 실패하는 단위 테스트 없이 프로덕션 코드를 작성하지 않는다.
2. **Green**: 테스트를 통과하는 최소한의 코드만 작성한다.
3. **Refactor**: 리팩토링 중에 기능을 추가하지 않는다.

테스트 구조: **Given-When-Then** 패턴.
테스트 명명: `test_should_[expected_behavior]_when_[condition]`

### 5.2 Modern Python
- **Type Hints**: `list[str]`, `dict`, `str | None` (내장 제네릭 사용)
- **Naming**: snake_case (변수/함수), PascalCase (클래스), UPPER_SNAKE_CASE (상수)
- **Immutability**: frozen dataclass 우선

### 5.3 Clean Code
- 작은 함수 (50줄 이하), 파일 (800줄 이하)
- 중첩 4단계 이하 (early return 활용)
- 단일 책임 원칙 (SRP)
- 최소 주석: 코드가 스스로 설명, 주석은 "Why"만

### 5.4 Documentation Language
모든 문서는 **한국어**로 작성. 기술 용어와 약어는 영어 유지 가능.

## 6. Test Protection Protocol

### 변경 전
1. `uv run pytest -v`로 베이스라인 확보
2. 모든 테스트 PASS 확인

### 변경 후
1. 전체 테스트 스위트 재실행
2. 변경 전 PASS → 변경 후 FAIL: 작업 중단, 사용자에게 보고
3. 자동 수정 금지: 사용자 승인 없이 실패한 테스트를 수정/삭제하지 않음

## 7. Git 규약

### Commit Message
```
<type>(<scope>): <subject>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Scopes: `agent`, `parser`, `llm`, `tools`, `pipeline`, `cli`, `report`, `deps`

### 예시
```
feat(tools): add ClinicalTrials.gov search tool
fix(agent): fallback to finish when LLM returns non-JSON
test(agent): add fallback scenario tests
```
