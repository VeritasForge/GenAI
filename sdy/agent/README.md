# Biomedical Research Agent

PubMed와 ClinicalTrials.gov를 활용하는 멀티스텝 ReAct 에이전트.
Claude Code CLI를 LLM 백엔드로 사용하며, Tool을 자율적으로 호출하여 약물 연구 질문에 답한다.

## 주요 기능

- **ReAct 패턴**: Observe → Think → Act 루프로 복잡한 질문을 단계적으로 해결
- **PubMed 검색**: 논문 검색 및 상세 정보 조회 (NCBI E-utilities API)
- **ClinicalTrials.gov 검색**: 임상시험 검색 (ClinicalTrials.gov v2 API)
- **구조화된 리포트**: 수집 데이터를 약물 리포트로 변환
- **CLI 인터페이스**: `ask`, `research`, `hello` 명령 지원

## 설치 및 실행

```bash
# 의존성 설치
uv sync --group dev

# smoke test
uv run agent hello

# 단일 질문
uv run agent ask "메트포르민의 최신 연구는?"

# 종합 리포트
uv run agent research "metformin"
```

## 프로젝트 구조

```
src/agent/
├── cli.py              # Typer CLI (ask, research, hello)
├── pipeline.py         # 오케스트레이션 (Tool 등록 + Agent 실행)
├── report.py           # 구조화된 리포트 생성
├── core/
│   ├── agent.py        # ReAct Agent Loop
│   ├── parser.py       # LLM 응답 JSON 파서
│   └── types.py        # AgentAction, AgentFinish, AgentResult
├── llm/
│   ├── client.py       # Claude CLI subprocess 래퍼
│   └── prompts.py      # 시스템 프롬프트 템플릿
└── tools/
    ├── registry.py     # Tool 등록/실행 레지스트리
    ├── types.py        # Article, Trial 데이터 클래스
    ├── pubmed.py       # PubMed API 클라이언트
    └── clinical_trials.py  # ClinicalTrials.gov API 클라이언트
```

## 아키텍처

```
User Query
    │
    ▼
  Agent (ReAct Loop)
    │
    ├─→ call_llm() ─→ Claude CLI ─→ JSON 응답
    │       │
    │       ▼
    │   parse_llm_response()
    │       │
    │       ├─ AgentAction → Tool 실행 → Observation → 루프 계속
    │       └─ AgentFinish → 답변 반환
    │
    └─→ Tools
         ├── search_pubmed()
         ├── fetch_article_details()
         └── search_trials()
```

핵심 설계:
- `pipeline.py`는 CLI 의존 없는 순수 로직 레이어
- `cli.py`는 파이프라인 함수를 호출하는 얇은 래퍼
- LLM이 JSON 대신 자연어로 응답하면 fallback으로 `AgentFinish` 처리

## 테스트

```bash
# 전체 테스트
uv run pytest -v

# 단일 파일
uv run pytest tests/test_agent.py -v

# 특정 테스트
uv run pytest -k "test_should_fallback" -v
```

## 린트

```bash
uv run ruff check src tests --fix
uv run ruff format src tests
```

## 기술 스택

- **Python 3.12+**, src layout, `uv` 패키지 매니저
- **LLM**: Claude Code CLI (`claude -p`)
- **API**: NCBI E-utilities (PubMed), ClinicalTrials.gov v2
- **HTTP**: httpx
- **CLI**: Typer
- **테스트**: pytest
- **린트**: ruff
- **빌드**: hatchling
