# Deep Research: RAG Serving Pipeline — 병목, 서비스 분리, 오케스트레이션, 대화 관리

## Executive Summary

RAG Serving Pipeline은 **순차적 파이프라인**이며, 각 단계가 독립적인 병목이 된다. 서비스 분리는 **MSA가 표준이지만 Monolith에서 시작**하여 점진적으로 분리하는 것이 권장된다. 각 컴포넌트 간 통신은 **트랜잭션이 아니라 동기 파이프라인 오케스트레이션**이며, 멀티턴 대화는 **외부 세션 스토어(Redis) + 토큰 버짓 관리**로 처리한다.

---

## Findings

### 1. 각 컴포넌트별 병목과 레이턴시 프로파일

- **확신도**: `[Confirmed]`
- **출처**: [RAGO — ISCA 2025](https://people.csail.mit.edu/suvinay/pubs/2025.rago.isca.pdf), [The New Stack — 5 RAG Bottlenecks](https://thenewstack.io/5-bottlenecks-impacting-rag-pipeline-efficiency-in-production/), [APXML — RAG Performance Bottlenecks](https://apxml.com/courses/optimizing-rag-for-production/chapter-1-production-rag-foundations/rag-performance-bottlenecks)

```
┌──────────────────────────────────────────────────────────────────────────┐
│              End-to-End 레이턴시 브레이크다운 (목표: 1.2~3초)              │
│                                                                          │
│  [API 수신]   [Query Embed]   [Vector Search]   [Rerank]     [LLM Gen]  │
│   ~1ms         ~10-50ms        ~10-100ms        ~200-500ms    ~1-3s     │
│   ├──┤         ├───────┤       ├────────┤       ├────────┤   ├───────┤  │
│                                                                          │
│   I/O         GPU/CPU         Vector DB         GPU          GPU        │
│   병목:        병목:           병목:             병목:         병목:       │
│   동시 연결    모델 로드        인덱스/샤딩       forward pass  토큰 생성   │
│                                                                          │
│   해결:        해결:           해결:             해결:         해결:       │
│   async       캐싱+배치       HNSW 튜닝         배치+양자화    SSE 스트림  │
│               +GPU가속        +Read Replica     +경량 모델    +KV 캐시    │
└──────────────────────────────────────────────────────────────────────────┘
```

| 컴포넌트 | 레이턴시 | 병목 원인 | 해소 전략 |
| -------- | -------- | --------- | --------- |
| **API Service** | ~1ms | 동시 연결 수, I/O 대기 | async/await (FastAPI), 커넥션 풀 |
| **Query Embedding** | 10~50ms | 모델 inference | 임베딩 캐시 (동일 쿼리 재계산 방지), GPU 배치 |
| **Vector DB Search** | 10~100ms | 인덱스 크기, 메타데이터 필터 | HNSW 파라미터 튜닝, Read Replica, 샤딩 |
| **Reranker** | 200~500ms | 문서당 full forward pass | GPU 필수, 배치 처리, 경량 모델 (TinyBERT 14M) |
| **LLM Generation** | 1~3s | 토큰 단위 생성 | SSE 스트리밍, KV 캐시, 양자화, vLLM |

**RAGO 논문의 핵심 발견**: 병목이 고정적이지 않다. 대형 LLM(70B+)에서는 LLM이 97% 병목이지만, 소형 모델(7B)에서는 retrieval이 50~75%를 차지한다. **프로파일링 없이 최적화하면 안 된다.**

---

### 2. 서비스 분리: MSA vs Monolith — "Start Mono, Extract Gradually"

- **확신도**: `[Confirmed]`
- **출처**: [Comet — Scalable LLM/RAG Inference Pipelines](https://www.comet.com/site/blog/llm-rag-inference-pipelines/), [DecodingML — Monolith vs Micro](https://decodingml.substack.com/p/monolith-vs-micro-the-1m-ml-design), [TDS — From Notebook to Microservices](https://towardsdatascience.com/the-journey-of-rag-development-from-notebook-to-microservices-cc065d0210ef/)

프로덕션 RAG의 서비스 분리는 **점진적**이다:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 1: Monolith (POC → 초기 프로덕션)                             │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              Single Service (FastAPI)                     │       │
│  │  [API] → [Embed] → [VectorDB Client] → [Rerank] → [LLM] │       │
│  │                   (모두 in-process)                       │       │
│  └─────────────────────────────────────────────────────────┘       │
│  장점: 빠른 개발, 단순한 배포, 디버깅 용이                           │
│  단점: GPU/CPU 리소스 경합, 개별 스케일링 불가                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  Phase 2: 최소 분리 (프로덕션 초기)                                  │
│                                                                     │
│  ┌──────────────────────┐    ┌──────────────────────┐              │
│  │  Orchestrator (CPU)   │    │   LLM Service (GPU)  │              │
│  │  [API + Embed +       │───→│   [vLLM / TGI]       │              │
│  │   Retrieve + Rerank]  │←───│                      │              │
│  └──────────────────────┘    └──────────────────────┘              │
│  LLM만 분리: GPU 리소스가 가장 비싸고 독립 스케일링 필요               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  Phase 3: 풀 MSA (대규모 트래픽)                                     │
│                                                                     │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   │
│  │ API/Orch  │──→│ Embedding │──→│ Reranker  │──→│ LLM       │   │
│  │ (CPU)     │   │ (GPU)     │   │ (GPU)     │   │ (GPU)     │   │
│  └─────┬─────┘   └───────────┘   └───────────┘   └───────────┘   │
│        │                                                           │
│        └──→ [Vector DB (Managed: Pinecone/Milvus/Weaviate)]        │
│        └──→ [Session Store (Redis)]                                │
│        └──→ [Cache (Redis/Memcached)]                              │
│  각 서비스 독립 스케일링, gRPC 내부 통신                               │
└─────────────────────────────────────────────────────────────────────┘
```

**핵심 원칙**: "Most AI systems don't need microservices until they're handling millions of requests." — Monolith에서 시작하여 **병목이 되는 컴포넌트부터** 하나씩 분리한다.

---

### 3. 오케스트레이션: 트랜잭션이 아니라 "동기 파이프라인"

- **확신도**: `[Confirmed]`
- **출처**: [Leanware — RAG Orchestration Services](https://www.leanware.co/insights/rag-orchestration-services), [Stack Overflow — RAG Architecture](https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/)

사용자가 궁금해하는 핵심: **"분리된 서비스들이 어떻게 transactional하게 동작하는가?"**

답: **트랜잭션이 아니다.** RAG Serving은 DB의 ACID 트랜잭션이 필요 없는 **읽기 전용 파이프라인**이다.

```
┌──────────────────────────────────────────────────────────────────┐
│                 Orchestrator의 동작 방식                           │
│                                                                  │
│  async def ask(query: str, session_id: str):                     │
│      # 1. 세션 히스토리 로드                                      │
│      history = await redis.get_history(session_id)               │
│                                                                  │
│      # 2. 쿼리 리라이팅 (멀티턴 시 필요)                           │
│      standalone_query = await rewrite_query(query, history)      │
│                                                                  │
│      # 3. 임베딩 (gRPC 호출 or in-process)                       │
│      query_vector = await embedding_service.embed(standalone_q)  │
│                                                                  │
│      # 4. Vector DB 검색                                         │
│      candidates = await vector_db.search(query_vector, top_k=20)│
│                                                                  │
│      # 5. Reranking (gRPC 호출)                                  │
│      reranked = await reranker_service.rerank(                   │
│          query=standalone_query,                                 │
│          documents=candidates,                                   │
│          top_n=5                                                 │
│      )                                                           │
│                                                                  │
│      # 6. 프롬프트 조립 (Orchestrator가 직접 수행)                 │
│      prompt = build_prompt(                                      │
│          system_prompt=SYSTEM_PROMPT,                             │
│          history=history[-5:],      # 최근 5턴                   │
│          context=reranked,          # reranked top-5 문서        │
│          query=query                                             │
│      )                                                           │
│                                                                  │
│      # 7. LLM 호출 + SSE 스트리밍                                 │
│      async for token in llm_service.stream(prompt):              │
│          yield token                                             │
│                                                                  │
│      # 8. 히스토리 저장                                           │
│      await redis.append_history(session_id, query, full_answer)  │
└──────────────────────────────────────────────────────────────────┘
```

**왜 트랜잭션이 불필요한가?**

| 일반 MSA (예: 주문 시스템) | RAG Serving Pipeline |
| ------------------------- | ------------------- |
| 데이터 변경 (Write) | **읽기 전용** (Read-only) |
| 여러 DB에 걸친 상태 변경 | 상태 변경 없음 (세션 저장 제외) |
| Saga / 2PC 필요 | 불필요 |
| 보상 트랜잭션 필요 | 실패 시 단순 재시도 또는 에러 반환 |
| 분산 락 필요 | 불필요 |

**실패 처리**: 파이프라인 중간에 실패하면? 단순히 에러를 클라이언트에 반환한다. 롤백할 "상태 변경"이 없기 때문이다. Vector DB 검색이 실패하면 → 재시도. Reranker가 실패하면 → 검색 결과를 그대로 LLM에 전달(graceful degradation). LLM이 실패하면 → 에러 반환.

---

### 4. 프롬프트 조립: Reranking 결과 + Query를 어떻게 LLM에 전달하는가

- **확신도**: `[Confirmed]`
- **출처**: [Prompt Engineering Guide — RAG](https://www.promptingguide.ai/research/rag), [Neo4j — Advanced RAG](https://neo4j.com/blog/genai/advanced-rag-techniques/)

Orchestrator가 **직접** 프롬프트를 조립한다. Reranker는 문서 순서만 정해주고, 최종 프롬프트는 Orchestrator의 책임이다.

```
┌──────────────────────────────────────────────────────────┐
│                 LLM에 전달되는 최종 프롬프트                │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │ [System Prompt] (~500 tokens)                     │    │
│  │ "당신은 사내 문서 기반 Q&A 어시스턴트입니다..."      │    │
│  ├──────────────────────────────────────────────────┤    │
│  │ [Conversation History] (~1,500 tokens)            │    │
│  │ User: 지난번에 물어본 퇴직금 계산 방법은?           │    │
│  │ Assistant: 퇴직금은 평균임금 × 근속연수...          │    │
│  │ User: 그럼 중간정산은?                             │    │
│  │ Assistant: 중간정산은 다음 조건에서...              │    │
│  ├──────────────────────────────────────────────────┤    │
│  │ [Retrieved Context] (~3,000 tokens)               │    │
│  │ --- 문서 1 (관련도 0.95) ---                       │    │
│  │ "퇴직금 중간정산 규정 제12조..."                    │    │
│  │ --- 문서 2 (관련도 0.91) ---                       │    │
│  │ "중간정산 신청 절차는..."                          │    │
│  │ --- 문서 3 (관련도 0.87) ---                       │    │
│  │ "예외 사항: 주택 구입 시..."                       │    │
│  ├──────────────────────────────────────────────────┤    │
│  │ [Current Query]                                   │    │
│  │ "중간정산 신청 기한이 언제까지야?"                   │    │
│  ├──────────────────────────────────────────────────┤    │
│  │ [Output Buffer] (~2,000 tokens 예약)              │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  총 토큰 버짓: ~7,000 / 32,000 (모델 한도)               │
└──────────────────────────────────────────────────────────┘
```

**토큰 버짓 공식**:
```
Available_Context = Model_Limit - (System_Prompt + Expected_Output)
History_Budget = Available_Context × 0.3
Retrieved_Context_Budget = Available_Context × 0.6
Query_Budget = Available_Context × 0.1
```

---

### 5. 멀티턴 대화 관리: 세션 스토어 + 컨텍스트 압축

- **확신도**: `[Confirmed]`
- **출처**: [Redis — LLM Session Memory](https://redis.io/docs/latest/develop/ai/redisvl/user_guide/session_manager/), [GetMaxim — Context Window Management](https://www.getmaxim.ai/articles/context-window-management-strategies-for-long-context-ai-agents-and-chatbots/), [Chitika — Handling Long Chat Histories](https://www.chitika.com/strategies-handling-long-chat-rag/)

```
┌──────────────────────────────────────────────────────────────────┐
│               멀티턴 대화 관리 아키텍처                             │
│                                                                  │
│  Client                                                          │
│    │ query + session_id                                          │
│    ▼                                                             │
│  ┌──────────────┐                                                │
│  │ Orchestrator  │                                               │
│  │              ├──→ [Redis: Session Store]                      │
│  │              │     ┌─────────────────────────────────┐       │
│  │              │     │ session:abc123                    │       │
│  │              │     │ ┌───────────────────────────────┐│       │
│  │              │     │ │ summary: "이전 대화 요약..."    ││       │
│  │              │     │ │ recent_turns: [                ││       │
│  │              │     │ │   {role: user, content: ...},  ││       │
│  │              │     │ │   {role: assistant, content: …}││       │
│  │              │     │ │   ... (최근 5턴)               ││       │
│  │              │     │ │ ]                              ││       │
│  │              │     │ │ entities: {퇴직금: ..., ...}   ││       │
│  │              │     │ │ total_tokens: 4,200            ││       │
│  │              │     │ └───────────────────────────────┘│       │
│  │              │     └─────────────────────────────────┘       │
│  │              │                                                │
│  │   토큰 체크: │                                                │
│  │   history_tokens > threshold?                                │
│  │     ├── No  → 최근 턴 그대로 사용                              │
│  │     └── Yes → Compaction/Summarization 실행                   │
│  └──────────────┘                                                │
└──────────────────────────────────────────────────────────────────┘
```

**컨텍스트 관리 전략 (우선순위순)**:

```
1. Raw (원본 유지)
   최근 5-7턴은 원본 그대로 보존
   ↓ (토큰 부족 시)

2. Compaction (가역적 압축)
   환경에 존재하는 중복 정보 제거
   예: 코드 블록 → "파일 X 참조" (필요 시 다시 읽기 가능)
   ↓ (여전히 부족 시)

3. Summarization (비가역적 요약)
   LLM으로 이전 대화를 요약
   최근 턴은 raw 유지 + 이전 턴은 요약으로 대체
   ↓ (극단적인 경우)

4. Semantic Retrieval of History
   대화 히스토리 자체를 벡터화하여 Redis/Vector DB에 저장
   현재 쿼리와 의미적으로 관련된 이전 턴만 선택적으로 로드
```

**Pre-Rot Threshold**: 모델이 1M 컨텍스트 윈도우라도 256k 이상에서 성능 저하 시작. API 에러가 나기 전에 **proactive하게** compaction/summarization을 트리거해야 한다.

**핵심 인사이트**: "Context Engineering is not about adding more context. It is about finding the **minimal effective context** required for the next step."

---

### 6. 서비스 간 통신 패턴

- **확신도**: `[Confirmed]`
- **출처**: [AWS — gRPC vs REST](https://aws.amazon.com/compare/the-difference-between-grpc-and-rest/), [Zen Van Riel — AI System Design Patterns 2026](https://zenvanriel.nl/ai-engineer-blog/ai-system-design-patterns-2026/)

| 구간 | 프로토콜 | 이유 |
| ---- | -------- | ---- |
| Client → API Gateway | **REST (HTTP)** | 브라우저 호환, SSE 스트리밍 |
| Orchestrator → Embedding Service | **gRPC** | 벡터 데이터 전송 효율, 내부 통신 |
| Orchestrator → Vector DB | **gRPC / REST** | DB 클라이언트 SDK에 따라 |
| Orchestrator → Reranker | **gRPC** | 대량 문서 전송, 저지연 |
| Orchestrator → LLM Service | **gRPC + SSE** | 토큰 스트리밍 |
| Orchestrator → Redis | **Redis Protocol** | 세션 읽기/쓰기 |

**Stateless 원칙**: 모든 서비스는 **stateless**. 세션 상태는 Redis에만 존재. 어떤 Orchestrator 인스턴스가 요청을 받아도 Redis에서 세션을 로드하여 동일하게 처리 가능 → 수평 스케일링 가능.

---

## 전체 아키텍처 종합

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Client (Browser/App)                          │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │ session_id + query + conversation_history (client-side)   │       │
│  └────────────────────────────┬─────────────────────────────┘       │
└───────────────────────────────┼──────────────────────────────────────┘
                                │ REST + SSE
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      API Gateway / LB                                │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  Orchestrator Service (CPU, Stateless)                │
│                                                                      │
│  1. Redis에서 세션 히스토리 로드                                       │
│  2. Query Rewriting (멀티턴 → standalone query)                      │
│  3. Semantic Cache 확인 → 히트 시 즉시 반환                           │
│  4. Embedding Service 호출 (query → vector)                          │
│  5. Vector DB 검색 (top-20)                                          │
│  6. Reranker Service 호출 (top-20 → top-5)                           │
│  7. Prompt 조립 (system + history + context + query)                 │
│  8. LLM Service 호출 + SSE 스트리밍                                   │
│  9. Redis에 히스토리 저장 + 토큰 카운트 업데이트                        │
│ 10. 토큰 threshold 초과 시 → Compaction/Summarization 트리거          │
│                                                                      │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐        │
│  │Embed   │  │VectorDB│  │Reranker│  │LLM     │  │Redis   │        │
│  │Service │  │(Managed│  │Service │  │Service │  │(Session│        │
│  │(GPU)   │  │ or     │  │(GPU)   │  │(GPU)   │  │ Store) │        │
│  │        │  │ Self)  │  │        │  │(vLLM)  │  │        │        │
│  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Edge Cases & Caveats

- **Reranker 장애 시 graceful degradation**: Reranker가 다운되면 Vector DB의 검색 결과를 그대로 LLM에 전달. 품질은 떨어지지만 서비스는 유지
- **LLM 컨텍스트 윈도우 초과**: 토큰 버짓을 미리 계산하지 않으면 LLM API가 에러 반환. Orchestrator가 prompt 조립 시 **반드시** 토큰 카운트 검증
- **클라이언트 측 히스토리 vs 서버 측 히스토리**: 클라이언트가 전체 히스토리를 보내면 payload가 커짐. 프로덕션에서는 **session_id만 보내고 서버(Redis)에서 히스토리를 관리**하는 것이 표준
- **GPU amplification**: Embed + Rerank + LLM 각각 GPU를 사용하므로, 요청 1건당 GPU 호출이 3회. 트래픽 증가 시 GPU 비용이 선형 이상으로 증가

## Contradictions Found

- **"Monolith 유지" vs "MSA 필수"**: 일부 출처는 "Monolith로 충분"이라 하고, 다른 출처는 "MSA가 표준"이라 함 → **해결**: 둘 다 맞다. 규모에 따라 다르며, "Start Mono, Extract Gradually"가 업계 합의. `[Confirmed]`

---

## Sources

1. [RAGO: Systematic Performance Optimization — ISCA 2025](https://people.csail.mit.edu/suvinay/pubs/2025.rago.isca.pdf) — 학술 논문
2. [5 Bottlenecks Impacting RAG Pipeline Efficiency — The New Stack](https://thenewstack.io/5-bottlenecks-impacting-rag-pipeline-efficiency-in-production/) — 기술 블로그
3. [RAG Performance Bottlenecks — APXML](https://apxml.com/courses/optimizing-rag-for-production/chapter-1-production-rag-foundations/rag-performance-bottlenecks) — 가이드
4. [Build a Scalable Inference Pipeline for LLMs and RAG — Comet](https://www.comet.com/site/blog/llm-rag-inference-pipelines/) — 기술 블로그
5. [Monolith vs Micro: The $1M ML Design — DecodingML](https://decodingml.substack.com/p/monolith-vs-micro-the-1m-ml-design) — 기술 블로그
6. [From Notebook to Microservices — TDS](https://towardsdatascience.com/the-journey-of-rag-development-from-notebook-to-microservices-cc065d0210ef/) — 기술 블로그
7. [RAG Orchestration Services Guide — Leanware](https://www.leanware.co/insights/rag-orchestration-services) — 가이드
8. [Context Window Management Strategies — GetMaxim](https://www.getmaxim.ai/articles/context-window-management-strategies-for-long-context-ai-agents-and-chatbots/) — 기술 블로그
9. [Redis — LLM Session Memory](https://redis.io/docs/latest/develop/ai/redisvl/user_guide/session_manager/) — 공식 문서
10. [Handling Long Chat Histories in RAG — Chitika](https://www.chitika.com/strategies-handling-long-chat-rag/) — 기술 블로그
11. [Context Engineering for AI Agents Part 2 — Phil Schmid](https://www.philschmid.de/context-engineering-part-2) — 1차 자료
12. [AI System Design Patterns 2026 — Zen Van Riel](https://zenvanriel.nl/ai-engineer-blog/ai-system-design-patterns-2026/) — 기술 블로그
13. [Designing Production-Ready RAG Pipelines — HackerNoon](https://hackernoon.com/designing-production-ready-rag-pipelines-tackling-latency-hallucinations-and-cost-at-scale) — 기술 블로그

---

## Research Metadata
- 검색 쿼리 수: 7 (일반 6 + SNS 1)
- 수집 출처 수: 13
- 출처 유형 분포: 공식 1, 학술 1, 블로그 9, 가이드 2, SNS 0
- 확신도 분포: Confirmed 6, Likely 0, Uncertain 0, Unverified 0
