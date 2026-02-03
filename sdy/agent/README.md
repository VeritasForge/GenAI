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

## 동작 원리

### ReAct Loop 전체 흐름

`core/agent.py`의 `Agent.run()` 메서드가 아래 루프를 실행한다.

```
User Query
    │
    ▼
messages = ["User: {query}"]          ← L42
    │
    ▼
┌─ for _ in range(max_steps): ──────────────────────────┐  ← L45
│                                                        │
│   prompt = "\n".join(messages)                          │  ← L46
│       │                                                │
│       ▼                                                │
│   raw_response = call_llm(prompt, system=system)        │  ← L47
│       │                                                │
│       ▼                                                │
│   parsed = parse_llm_response(raw_response)             │  ← L50
│       │                                                │
│       ├─ AgentFinish ──→ return AgentResult (종료)       │  ← L54-59
│       │                                                │
│       └─ AgentAction                                   │
│              │                                         │
│              ▼                                         │
│         observation = _execute_tool(parsed)              │  ← L61
│              │                                         │
│              ▼                                         │
│         messages.append("Assistant: {raw_response}")    │  ← L64
│         messages.append("Observation: {observation}")   │  ← L65
│              │                                         │
│              └─→ 다음 루프 반복                           │
│                                                        │
└────────────────────────────────────────────────────────┘
    │
    ▼ (max_steps 초과)
raise MaxStepsExceededError                               ← L67-69
```

### 실제 실행 예시 — "메트포르민 연구" 질문

**Step 0**: messages 초기 상태

```
messages = ["User: 메트포르민 연구"]
```

→ LLM 응답:

```json
{"action": "tool", "tool": "search_pubmed", "args": {"query": "metformin"}}
```

→ App이 `search_pubmed("metformin")` 실행 → `"Found 3 articles..."`

---

**Step 1**: messages에 이전 결과가 누적된 상태

```
messages = [
    "User: 메트포르민 연구",
    "Assistant: {\"action\":\"tool\",\"tool\":\"search_pubmed\",\"args\":{\"query\":\"metformin\"}}",
    "Observation: Found 3 articles..."
]
```

→ LLM이 충분한 정보가 있다고 판단:

```json
{"action": "finish", "answer": "메트포르민 관련 3개 논문을 찾았습니다..."}
```

→ `AgentResult` 반환, 루프 종료

### 프롬프트 구성 — LLM에게 전달되는 전체 내용

`Agent.run()`은 두 가지를 LLM에 전달한다 (L39-47):

```
┌─ system prompt (--append-system-prompt) ─────────────┐
│ "You are a biomedical research assistant..."          │
│ + Tool 목록 (registry.get_prompt_description())       │
│ + 응답 형식: {"action":"tool",...} 또는 {"action":"finish",...} │
│ + 대화 규칙: User/Assistant/Observation 역할 설명      │
└──────────────────────────────────────────────────────┘

┌─ prompt ("\n".join(messages)) ────────────────────────┐
│ User: 메트포르민 연구                                   │
│ Assistant: {"action":"tool","tool":"search_pubmed",...} │
│ Observation: Found 3 articles...                       │
│ (매 루프마다 Assistant + Observation이 추가됨)            │
└──────────────────────────────────────────────────────┘
```

### 핵심 설계 포인트

| 설계 결정 | 설명 |
|-----------|------|
| **LLM은 판단만, App이 실행** | LLM은 JSON으로 "다음 행동"을 지시하고, 실제 API 호출은 `_execute_tool()`이 수행 |
| **messages 누적** | 매 스텝마다 Assistant 응답 + Observation을 messages에 추가하여 LLM에게 전체 맥락 전달 |
| **Fallback 처리** | Claude CLI가 JSON 대신 자연어로 응답하면 `AgentFinish(answer=raw_response)`로 처리 (L51-52) |
| **파이프라인 분리** | `pipeline.py`는 CLI 의존 없는 순수 로직, `cli.py`는 파이프라인을 호출하는 얇은 래퍼 |

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
