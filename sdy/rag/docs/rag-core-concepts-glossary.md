# Deep Research: RAG 시스템 핵심 개념 총정리

## Executive Summary

Orchestrator는 파이프라인 조율자 역할이 맞다. 각 기술 개념(forward pass, HNSW 튜닝, 샤딩, KV 캐시, 양자화, vLLM/TGI)을 아래에 정리한다. **세션 관리에 대해서는 사용자의 직관이 맞다** — OpenAI API 자체는 클라이언트가 전체 messages를 보내는 stateless 방식이다. 서버 측 Redis 관리는 "프로덕션 편의를 위한 선택"이지 유일한 방법이 아니다.

---

## 0. RAG 시스템 용어 정리

- **확신도**: `[Confirmed]`

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RAG 파이프라인 용어 맵                            │
│                                                                     │
│  ┌─────────────────── Ingestion (Offline) ──────────────────┐      │
│  │                                                           │      │
│  │  Parsing → Chunking → Embedding → Indexing               │      │
│  │  (파싱)   (청킹)     (임베딩)    (인덱싱/저장)            │      │
│  └───────────────────────────────────────────────────────────┘      │
│                                                                     │
│  ┌─────────────────── Serving (Online) ─────────────────────┐      │
│  │                                                           │      │
│  │  Query        → Retrieval    → Reranking   → Generation  │      │
│  │  (질의 처리)    (검색/회수)    (재순위화)     (답변 생성)  │      │
│  │                                                           │      │
│  │  세부:                                                    │      │
│  │  ┌────────┐   ┌────────────┐  ┌─────────┐  ┌──────────┐│      │
│  │  │Query   │   │Retrieval   │  │Reranking│  │Generation││      │
│  │  │Rewrite │   │= Embed     │  │= Cross- │  │= LLM     ││      │
│  │  │= 멀티턴│   │+ Vector    │  │  Encoder │  │  호출 +  ││      │
│  │  │ 질의를 │   │  Search    │  │  scoring │  │  토큰    ││      │
│  │  │ 독립   │   │+ Metadata  │  │+ top-N   │  │  생성    ││      │
│  │  │ 질의로 │   │  Filtering │  │  선별    │  │         ││      │
│  │  │ 변환   │   │= "Recall"  │  │         │  │         ││      │
│  │  └────────┘   └────────────┘  └─────────┘  └──────────┘│      │
│  └───────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

| 용어 | 의미 | 해당 단계 |
| ---- | ---- | --------- |
| **Retrieval** | 쿼리 임베딩 + Vector DB 검색 + 메타데이터 필터링 **전체** | Serving 2단계 |
| **Recall** | Retrieval과 동의어. 또는 "관련 문서를 빠뜨리지 않는 능력" (지표) | Serving 2단계 |
| **Reranking** | 검색된 후보를 Cross-Encoder로 재순위화 | Serving 3단계 |
| **Generation** | LLM에 프롬프트를 보내 답변 생성 | Serving 4단계 |
| **Augmentation** | 검색된 문서를 프롬프트에 삽입하는 행위 | Serving 3~4단계 사이 |
| **Orchestration** | 전체 파이프라인을 조율하는 것 | 전 단계 |
| **Ingestion** | 문서를 파싱→청킹→임베딩→저장하는 오프라인 작업 | Offline |
| **Indexing** | 벡터를 Vector DB에 저장하고 인덱스를 구축하는 것 | Offline 마지막 |

**RAGO 논문에서 "Retrieval이 50~75% 병목"**이라 할 때, 이는 **쿼리 임베딩 + ANN 검색 + (경우에 따라) 메타데이터 필터링**을 합산한 것이다.

---

## 1. Forward Pass란?

- **확신도**: `[Confirmed]`

```
┌─────────────────────────────────────────────────────────────┐
│                    Forward Pass (순전파)                      │
│                                                             │
│  신경망에서 입력 → 출력까지 한 번 계산하는 것                  │
│                                                             │
│  Input                                                      │
│  "중간정산 기한은?"  ──→  [Layer 1] ──→ [Layer 2] ──→ ...   │
│  +                        (행렬곱)     (행렬곱)              │
│  "퇴직금 규정 제12조..."  (활성화)     (활성화)              │
│                                ──→ [Output Layer]           │
│                                     관련성 점수: 0.92        │
│                                                             │
│  이 한 번의 계산 = 1 Forward Pass                            │
│                                                             │
│  Cross-Encoder Reranking에서:                                │
│  문서 20개 × 1 forward pass = 20번의 full Transformer 계산   │
│  → 이것이 200~500ms 걸리는 이유                              │
└─────────────────────────────────────────────────────────────┘
```

학습(training)에서는 forward pass 후 **backward pass**(역전파, 기울기 계산)가 따라오지만, 추론(inference)에서는 **forward pass만** 수행한다. Reranker가 "문서당 1 forward pass"라는 것은 (query + doc) 쌍을 Transformer 전체 레이어에 통과시키는 계산을 문서마다 해야 한다는 뜻이다.

---

## 2. HNSW 파라미터 튜닝

- **확신도**: `[Confirmed]`
- **출처**: [Milvus HNSW Docs](https://milvus.io/docs/hnsw.md), [OpenSearch HNSW Guide](https://opensearch.org/blog/a-practical-guide-to-selecting-hnsw-hyperparameters/)

```
┌──────────────────────────────────────────────────────────────────┐
│                    HNSW 그래프 구조                                │
│                                                                  │
│  Layer 2 (sparse):    A ─────────── D                           │
│                                                                  │
│  Layer 1 (medium):    A ── B ────── D ── E                      │
│                                                                  │
│  Layer 0 (dense):     A ── B ── C ── D ── E ── F ── G          │
│                        │         │         │                     │
│                        └────┬────┘         │                     │
│                             │              │                     │
│                          M개의 연결        M개의 연결             │
│                                                                  │
│  검색: 상위 레이어에서 대략적 위치 → 하위로 내려가며 정밀 탐색    │
└──────────────────────────────────────────────────────────────────┘
```

| 파라미터 | 의미 | 영향 | 기본값 |
| -------- | ---- | ---- | ------ |
| **M** | 노드당 최대 연결 수 | 높을수록: recall↑, 메모리↑, 빌드 시간↑ | 16 |
| **efConstruction** | 인덱스 빌드 시 탐색 깊이 | 높을수록: 그래프 품질↑, 빌드 시간↑ | 64~200 |
| **efSearch** | 검색 시 탐색 깊이 | 높을수록: recall↑, 검색 시간↑ | 100 |

**실전 튜닝 가이드**:

| 시나리오 | M | efConstruction | efSearch | recall | 레이턴시 |
| -------- | -- | -------------- | -------- | ------ | -------- |
| 고정확도 (추천 시스템) | 24 | 400 | 200~500 | ~98% | ~5ms |
| 저지연 (실시간 검색) | 12 | 100~200 | 50~100 | ~85% | ~1ms |
| **기본 시작점** | **16** | **200** | **100** | ~90% | ~2ms |

**메모리 영향**: M만 메모리에 영향을 준다 (연결이 많으면 그래프가 커짐). efConstruction과 efSearch는 메모리에 영향 없고 계산 시간만 변화.

---

## 3. 샤딩: 원리와 Hot Spot 문제

- **확신도**: `[Confirmed]`
- **출처**: [Milvus — Sharding and Replication](https://milvus.io/ai-quick-reference/how-do-distributed-vector-databases-handle-sharding-and-replication), [Medium — Deep Dive into Sharding](https://medium.com/startlovingyourself/from-hot-keys-to-rebalancing-a-deep-dive-into-sharding-dcb48c69bab7)

```
┌──────────────────────────────────────────────────────────────────┐
│              Hash-Based Sharding 원리                             │
│                                                                  │
│  문서 ID → hash(doc_id) → shard_number                          │
│                                                                  │
│  doc_001 → hash("001") % 3 = 0 → Shard 0                       │
│  doc_002 → hash("002") % 3 = 1 → Shard 1                       │
│  doc_003 → hash("003") % 3 = 2 → Shard 2                       │
│  doc_004 → hash("004") % 3 = 1 → Shard 1                       │
│                                                                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                         │
│  │ Shard 0  │  │ Shard 1  │  │ Shard 2  │                       │
│  │ Node A   │  │ Node B   │  │ Node C   │                       │
│  │ doc_001  │  │ doc_002  │  │ doc_003  │                       │
│  │ ...      │  │ doc_004  │  │ ...      │                       │
│  └─────────┘  └─────────┘  └─────────┘                         │
│                                                                  │
│  검색 시: 모든 샤드에 병렬 쿼리 → 각 샤드 top-K → 머지           │
└──────────────────────────────────────────────────────────────────┘
```

### Hot Spot 문제

```
문제: 특정 샤드에 인기 데이터가 집중

  Shard 0: (90% 트래픽) ← Hot Spot!
  Shard 1: (5% 트래픽)
  Shard 2: (5% 트래픽)

원인:
  - 특정 카테고리/테넌트의 문서가 한 샤드에 몰림
  - 인기 문서가 특정 샤드에 집중
  - 불균등한 hash 분포
```

### 해결 전략

| 전략 | 방법 | 적용 |
| ---- | ---- | ---- |
| **Consistent Hashing** | 해시 링에 가상 노드(virtual nodes) 배치. 노드 추가/제거 시 O(1/N)만 재배치 | 동적 스케일링 |
| **Virtual Nodes** | 물리 노드 1개에 가상 노드 100~200개 매핑. 더 균등한 분산 | hot spot 완화 |
| **Dynamic Rebalancing** | 모니터링 후 부하 높은 샤드의 데이터를 자동 이동 | Milvus, Weaviate 내장 |
| **Metadata-Based Routing** | 카테고리/테넌트 기반으로 의도적 분산 | 멀티테넌트 |
| **Read Replica** | hot 샤드에 읽기 복제본 추가 | 읽기 부하 분산 |

---

## 4. KV 캐시 (KV Cache)

- **확신도**: `[Confirmed]`
- **출처**: [Sebastian Raschka — KV Cache from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms), [NVIDIA — LLM Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

```
┌──────────────────────────────────────────────────────────────────┐
│                    KV 캐시 없이 (느림)                             │
│                                                                  │
│  토큰 1 생성: attention("나는")           → K,V 계산             │
│  토큰 2 생성: attention("나는 오늘")       → K,V 전부 재계산!     │
│  토큰 3 생성: attention("나는 오늘 점심")  → K,V 전부 재계산!     │
│  ...반복할수록 계산량 O(n²)으로 폭증                              │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                    KV 캐시 있으면 (빠름)                           │
│                                                                  │
│  토큰 1: attention("나는")           → K₁,V₁ 계산 + 캐시 저장   │
│  토큰 2: attention("오늘")           → K₂,V₂만 새로 계산        │
│          캐시에서 K₁,V₁ 재사용       → [K₁,K₂], [V₁,V₂]        │
│  토큰 3: attention("점심")           → K₃,V₃만 새로 계산        │
│          캐시에서 K₁₂,V₁₂ 재사용     → [K₁,K₂,K₃], [V₁,V₂,V₃] │
│                                                                  │
│  → 이전 토큰의 K,V를 다시 계산하지 않아 O(n)으로 감소             │
└──────────────────────────────────────────────────────────────────┘
```

**문제**: KV 캐시가 GPU 메모리를 많이 차지. LLaMA-13B에서 시퀀스 1개당 최대 **1.7GB**. 기존 시스템은 메모리의 60~80%가 단편화로 낭비.

**해결**: **PagedAttention** — KV 캐시를 OS의 가상 메모리처럼 고정 크기 블록("페이지")으로 분할하여 비연속적으로 관리. 낭비율 4% 미만.

---

## 5. 양자화 (Quantization)

- **확신도**: `[Confirmed]`

```
┌──────────────────────────────────────────────────────────────────┐
│                    양자화 (Quantization)                          │
│                                                                  │
│  원래: 모델 가중치가 FP32 (32비트 부동소수점)                     │
│                                                                  │
│  FP32:  3.141592653589793...  (32bit, 높은 정밀도)              │
│  FP16:  3.14159...            (16bit, 절반 크기)                │
│  INT8:  3                     (8bit, 1/4 크기)                  │
│  INT4:  3                     (4bit, 1/8 크기)                  │
│                                                                  │
│  70B 모델 메모리 요구량:                                         │
│  FP32: ~280GB → GPU 불가                                        │
│  FP16: ~140GB → A100 80GB × 2                                  │
│  INT8:  ~70GB → A100 80GB × 1                                  │
│  INT4:  ~35GB → A100 40GB × 1                                  │
│                                                                  │
│  Trade-off:                                                      │
│  정밀도 ◄────────────────────────► 속도/메모리                    │
│  FP32      FP16      INT8     INT4                              │
│  최고정밀   ±0.1%↓    ±1-2%↓   ±3-5%↓                           │
└──────────────────────────────────────────────────────────────────┘
```

정밀도를 약간 희생하여 **메모리 절감 + 속도 향상**을 얻는 기법. "FP8 대비 FP16에서 최대 2배 속도 향상" (NVIDIA).

---

## 6. vLLM과 TGI

- **확신도**: `[Confirmed]`
- **출처**: [Modal — vLLM vs TGI](https://modal.com/blog/vllm-vs-tgi-article), [arXiv — vLLM vs TGI Comparison](https://arxiv.org/abs/2511.17593)

| 비교 기준 | vLLM | TGI (HuggingFace) |
| --------- | ---- | ------------------ |
| 핵심 기술 | **PagedAttention** | Continuous Batching + Prefix Caching |
| 최적 시나리오 | 고동시성 배치 처리 | 인터랙티브 저지연 |
| 처리량 | TGI 대비 최대 **24x** (고동시성) | vLLM 대비 낮음 (v3에서 격차 축소) |
| GPU 활용률 | 85~92% | 68~74% |
| 동시 요청 포화점 | 100~150 | 50~75 |
| 프로덕션 기능 | 기본적 | OpenTelemetry, Prometheus 내장 |
| 개발사 | UC Berkeley | HuggingFace |

**왜 vLLM이 "해소 전략"인가?**

LLM 추론의 최대 병목은 **GPU 메모리 낭비**(KV 캐시 단편화)와 **배치 비효율**이다. vLLM의 PagedAttention이 메모리 낭비를 4% 미만으로 줄이고, Continuous Batching이 GPU 유휴 시간을 제거하여 같은 GPU에서 **수~수십 배** 높은 처리량을 달성한다.

**TGI (Text Generation Inference)**: HuggingFace가 만든 프로덕션 LLM 서빙 툴킷. vLLM과 같은 목적이지만 엔터프라이즈 운영 기능(모니터링, 텔레메트리)이 더 강하고, TGI v3에서 긴 컨텍스트에 대해 vLLM 대비 **13x 빠른** 추론을 달성하여 격차를 크게 줄였다.

---

## 7. 세션 관리: 클라이언트 vs 서버 — 사용자의 직관이 맞다

- **확신도**: `[Confirmed]`
- **출처**: [nityesh.com — LLMs are Stateless](https://nityesh.com/llms-are-stateless-each-msg-you-send-is-a-fresh-start-even-if-its-in-a-thread/), [Redis — Session Memory](https://redis.io/docs/latest/develop/ai/redisvl/user_guide/session_manager/), [Arize — Memory and State in LLM Apps](https://arize.com/blog/memory-and-state-in-llm-applications/)

사용자의 관찰이 정확하다. **실제로 두 가지 패턴이 모두 쓰인다:**

```
┌──────────────────────────────────────────────────────────────────┐
│          패턴 A: Client-Side (OpenAI API 방식)                    │
│                                                                  │
│  Client가 전체 messages 배열을 매 요청마다 보냄                   │
│                                                                  │
│  Client                          Server (Stateless)              │
│  ┌──────────────────┐            ┌──────────────────┐           │
│  │ messages = [      │            │                  │           │
│  │   {system: ...},  │───POST──→│  LLM.generate(   │           │
│  │   {user: Q1},     │            │    messages      │           │
│  │   {assistant: A1},│            │  )               │           │
│  │   {user: Q2},     │            │                  │           │
│  │   {assistant: A2},│            │  → 응답 반환     │           │
│  │   {user: Q3}      │            │  → 상태 저장 X   │           │
│  │ ]                 │            └──────────────────┘           │
│  └──────────────────┘                                           │
│                                                                  │
│  ✅ 완전 Stateless                                               │
│  ✅ 서버 확장 용이                                                │
│  ❌ 대화가 길어지면 payload가 거대해짐                             │
│  ❌ 매 요청마다 이전 토큰 비용 재지불                               │
│  ❌ 컨텍스트 윈도우 초과 시 클라이언트가 truncation 처리해야 함     │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│          패턴 B: Server-Side (ChatGPT 웹앱 / 프로덕션 RAG)       │
│                                                                  │
│  Client는 session_id + 현재 query만 보냄                         │
│  서버가 Redis/DB에서 히스토리를 로드                               │
│                                                                  │
│  Client                          Server                          │
│  ┌──────────────────┐            ┌──────────────────┐           │
│  │ {                 │            │  1. Redis에서     │           │
│  │   session_id: abc │───POST──→│     히스토리 로드  │           │
│  │   query: "Q3"     │            │  2. 히스토리 +    │           │
│  │ }                 │            │     query 조합    │           │
│  └──────────────────┘            │  3. LLM 호출      │           │
│                                  │  4. 응답 저장     │           │
│                                  └──────────────────┘           │
│                                                                  │
│  ✅ 작은 payload (query + session_id만)                           │
│  ✅ 서버가 summarization/compaction 자동 처리                     │
│  ✅ 크로스 디바이스 대화 연속성                                    │
│  ✅ RAG 파이프라인 통합 용이 (서버가 context 조립)                 │
│  ❌ 서비스는 stateless지만 **외부 상태 저장소(Redis) 의존**        │
│  ❌ Redis 장애 시 히스토리 유실 가능                               │
└──────────────────────────────────────────────────────────────────┘
```

### 실제 서비스별 패턴

| 서비스 | 패턴 | 이유 |
| ------ | ---- | ---- |
| **OpenAI Chat Completions API** | Client-Side (A) | API는 stateless. 클라이언트가 messages 전체를 보냄 |
| **ChatGPT 웹앱** | Server-Side (B) | 웹앱이 서버에 대화 저장. 브라우저 바꿔도 대화 유지 |
| **Claude 웹앱** | Server-Side (B) | 동일 |
| **프로덕션 RAG 서비스** | **하이브리드** | 아래 설명 |

### 프로덕션 RAG에서 서버 측 관리를 하는 이유

패턴 A(클라이언트가 전체 히스토리를 보내는 방식)는 **단순 LLM 챗봇**에서는 잘 작동하지만, **RAG 시스템에서는 한계**가 있다:

1. **서버에서만 할 수 있는 작업이 있다**: Query Rewriting, Semantic Cache 확인, 이전 검색 결과 재활용 — 이런 작업들은 서버가 히스토리를 알아야 가능
2. **RAG 컨텍스트 = 대화 히스토리 + 검색 결과**: 클라이언트는 검색 결과(retrieved documents)를 가지고 있지 않다. 서버가 조립해야 한다
3. **토큰 비용 최적화**: 서버가 히스토리를 관리하면 summarization/compaction을 자동으로 수행하여 토큰 비용 절감
4. **보안**: 민감한 문서 컨텍스트가 클라이언트로 왕복하지 않음

**하지만 이것이 "stateless 위반"은 아니다.** 서비스 자체는 stateless이고, 상태는 **외부 저장소(Redis)**에 있다. 어떤 Orchestrator 인스턴스가 요청을 받아도 Redis에서 세션을 로드하여 동일하게 처리 가능 → **수평 스케일링에 문제 없다.** 이것은 일반 웹 서비스에서 세션을 Redis에 저장하는 것과 동일한 패턴이다.

```
"Stateless Service + External State Store"
= 업계 표준 패턴 (JWT + Redis Session과 동일한 원리)
= 서비스 인스턴스 자체에는 상태 없음
= 상태는 Redis/DB에 중앙 관리
= 어떤 인스턴스든 동일한 요청 처리 가능
```

---

## Orchestrator 역할 확인

맞다. Orchestrator는 **비즈니스 로직의 조율자**이다:

1. 각 서비스(Embedding, Vector DB, Reranker, LLM)를 **순차적으로 호출**
2. 이전 단계의 **출력을 다음 단계의 입력으로 전달**
3. **에러 처리** (재시도, graceful degradation, 에러 반환)
4. **프롬프트 조립** (히스토리 + 검색 결과 + 쿼리 결합)
5. **세션 관리** (Redis 읽기/쓰기)
6. **캐시 확인** (Semantic Cache 히트 시 파이프라인 스킵)

Orchestrator 자체는 CPU만 사용하는 가벼운 서비스이며, 무거운 계산은 모두 하위 GPU 서비스에 위임한다.

---

## Edge Cases & Caveats

- **클라이언트 측 히스토리가 더 적합한 경우**: 간단한 챗봇, 단발성 Q&A, 개인 데이터 보호가 중요한 경우 → 패턴 A가 더 낫다
- **서버 측 히스토리가 필수인 경우**: RAG(검색 결과를 서버가 조립해야 하므로), 멀티디바이스 대화 연속성, 관리자 감사 로그 필요 시
- **하이브리드**: 최근 N턴은 클라이언트가 보내고, 전체 히스토리는 서버가 관리하는 방식도 있음

## Contradictions Found

- **"서버 측이 표준" vs "클라이언트 측이 표준"**: 둘 다 맞다. API 레벨에서는 클라이언트 측(OpenAI 방식)이 표준이고, **애플리케이션 레벨**에서는 서버 측(Redis)이 표준이다. 실제로 OpenAI API를 쓰는 앱도 서버에서 messages 배열을 관리하고 API에 보내는 것이 일반적이다. `[Confirmed]`

---

## Sources

1. [Milvus — HNSW Parameters](https://milvus.io/ai-quick-reference/what-are-the-key-configuration-parameters-for-an-hnsw-index-such-as-m-and-efconstructionefsearch-and-how-does-each-influence-the-tradeoff-between-index-size-build-time-query-speed-and-recall) — 공식 문서
2. [OpenSearch — Practical Guide to HNSW Hyperparameters](https://opensearch.org/blog/a-practical-guide-to-selecting-hnsw-hyperparameters/) — 공식 블로그
3. [Modal — vLLM vs TGI](https://modal.com/blog/vllm-vs-tgi-article) — 벤치마크
4. [arXiv — vLLM vs TGI Performance Study](https://arxiv.org/abs/2511.17593) — 학술 논문
5. [Sebastian Raschka — KV Cache from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) — 1차 자료
6. [NVIDIA — LLM Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) — 공식 블로그
7. [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 1차 자료
8. [Milvus — Sharding and Replication](https://milvus.io/ai-quick-reference/how-do-distributed-vector-databases-handle-sharding-and-replication) — 공식 문서
9. [Medium — Deep Dive into Sharding](https://medium.com/startlovingyourself/from-hot-keys-to-rebalancing-a-deep-dive-into-sharding-dcb48c69bab7) — 기술 블로그
10. [nityesh.com — LLMs are Stateless](https://nityesh.com/llms-are-stateless-each-msg-you-send-is-a-fresh-start-even-if-its-in-a-thread/) — 기술 블로그
11. [Redis — LLM Session Memory](https://redis.io/docs/latest/develop/ai/redisvl/user_guide/session_manager/) — 공식 문서
12. [Arize — Memory and State in LLM Apps](https://arize.com/blog/memory-and-state-in-llm-applications/) — 기술 블로그
13. [HuggingFace KV Caching Explained](https://huggingface.co/blog/not-lain/kv-caching) — 공식 블로그

---

## Research Metadata
- 검색 쿼리 수: 5 (일반 5 + SNS 0)
- 수집 출처 수: 13
- 출처 유형 분포: 공식 5, 학술 1, 1차 자료 2, 블로그 5, SNS 0
- 확신도 분포: Confirmed 8, Likely 0, Uncertain 0, Unverified 0
