# 프로덕션 RAG 아키텍처 설계 가이드

> 현재 `sdy/rag/` 프로젝트를 프로덕션 환경에 올린다고 가정했을 때의 아키텍처 분석 및 설계 문서.

---

## 목차

1. [전체 아키텍처 개요](#1-전체-아키텍처-개요)
2. [Layer 1: Data Ingestion Pipeline (Offline)](#2-layer-1-data-ingestion-pipeline-offline)
3. [Layer 2: Query/Serving Pipeline (Online)](#3-layer-2-queryserving-pipeline-online)
4. [Layer 3: Cross-Cutting Concerns](#4-layer-3-cross-cutting-concerns)
5. [심화 Q&A](#5-심화-qa)
   - [5.1 Response Cache (Semantic Cache)](#51-response-cache-semantic-cache)
   - [5.2 병목 해소 전략 — Scale Up vs Scale Out](#52-병목-해소-전략--scale-up-vs-scale-out)
   - [5.3 SSE Streaming 동작 방식](#53-sse-streaming-동작-방식)
6. [현재 코드베이스와의 갭 분석](#6-현재-코드베이스와의-갭-분석)
7. [Sources](#7-sources)

---

## 1. 전체 아키텍처 개요

프로덕션 RAG 시스템은 크게 **Ingestion Pipeline(Offline)**과 **Serving Pipeline(Online)**으로 분리된다. 이것은 업계 표준이며, 각 파이프라인은 독립적으로 스케일링하고 운영할 수 있다.

```
┌─────────────────────────────────────────────────────────┐
│                    Data Ingestion (Offline)              │
│                                                         │
│  [Admin UI / Batch / Event] → [Queue] → [ETL Pipeline]  │
│       → [Parse → Chunk → Embed → Vector DB]             │
│       + [Metadata Registry + Version Manager]           │
│       + [Dedup + Change Detection]                      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  Query Serving (Online)                  │
│                                                         │
│  [Client] → [API Gateway + LB]                          │
│    → [FastAPI async] → [Cache Check]                    │
│    → [Retriever (Hybrid Search)]                        │
│    → [Reranker] → [LLM Generator]                       │
│    → [SSE Streaming Response]                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                Cross-Cutting Concerns                   │
│                                                         │
│  [Monitoring/Logging] [Security/PII] [Feedback Loop]    │
└─────────────────────────────────────────────────────────┘
```

### 주요 설계 원칙

- **Modular RAG**: 각 컴포넌트(retriever, generator, orchestration)를 분리하여 독립적으로 업데이트/디버깅
- **Hybrid Search**: 벡터 검색(의미 유사) + 키워드 검색(정확 매칭) 결합
- **RAGOps**: 모든 RAG 컴포넌트(검색 소스, retriever, generator)에 걸쳐 쿼리, 응답, 사용자 피드백, 컴포넌트 입출력을 로깅

---

## 2. Layer 1: Data Ingestion Pipeline (Offline)

```
[Data Sources] → [Ingestion Queue (Kafka/SQS)] → [Parser] → [Chunker] → [Embedder] → [Vector DB]
                                                                                          ↑
                                                          [Metadata Registry] ← [Version Manager]
```

### 2.1 트리거 방식

실제 프로덕션에서는 3가지 트리거 방식을 혼용한다:

| 방식 | 사용 시점 | 기술 스택 예시 |
|------|----------|--------------|
| Admin UI | 수동 업로드, 즉시 반영 필요 | Django Admin, FastAPI + 별도 UI |
| Batch (스케줄) | 주기적 크롤링/동기화 | Airflow, Celery Beat, Cron |
| Event-Driven | 소스 시스템 변경 즉시 반영 | Kafka, Webhook, CDC(Change Data Capture) |

### 2.2 Ingestion Pipeline 핵심 단계

1. **파싱**: 다양한 포맷(PDF, Word, HTML) 처리, OCR/[NLP 추출](nlp-metadata-extraction-guide.md), 테이블 인식(Table Transformer)
2. **메타데이터 추출**: author, date, tags 등 필터링용 메타데이터 — [NLP/NER 기반 메타데이터 추출 가이드](nlp-metadata-extraction-guide.md) 참조
3. **청킹**: 도메인 인식 청킹 — 코드는 "function-based", 텍스트는 "paragraph-level"
4. **임베딩 생성**: 고차원 벡터로 변환
5. **벡터 DB 저장**: 빠른 검색/조회를 위한 벡터 데이터베이스에 적재

### 2.3 버전 관리 & 중복 방지

#### Document Versioning

- **Document-level version tracking**: 각 문서의 메타데이터 레지스트리에 버전 번호 추적
- 문서 업데이트 시 버전 증가 (예: `financial_report_2024.pdf` v1 → v2)
- 두 버전 모두 초기에 Vector DB에 저장, 버전 메타데이터로 태깅
- 현재 버전을 프로덕션에서 활성화, 이전 1~2개 버전은 롤백용 유지
- 검증 기간(7~14일) 후 이전 버전 퍼지하여 인덱스 비대화 방지
- 사용된 임베딩 모델 버전도 함께 추적

#### Deduplication

- 고급 노이즈 필터링(중복 감지, 시맨틱 중복 제거)으로 관련 데이터만 인제스트
- 기본 중복 제거는 첫 번째 결과를 임의 선택
- 개선 방안: "최상" 버전 선택 로직 (최근 업데이트, 발행 상태, 가장 권위 있는 소스)
- MinHash 모델의 피처화 단계와 해시 테이블 수 실험으로 매칭 결과 개선

#### Incremental Indexing

- 전체 재인덱싱 대신 **증분 인덱싱** 도입
- 문서 버전과 업데이트 타임스탬프를 추적
- 문서 변경 시 수정된 청크만 재임베딩 후 벡터 스토어 업데이트
- 비용이 큰 전체 재인덱스 작업 제거

#### Content-based Change Detection

- 해시 기반 변경 감지로 실제 변경된 문서만 재처리
- 메타데이터 레지스트리에서 Vector DB 내용 추적
- 벡터 데이터베이스의 네이티브 증분 삽입 지원 활용

### 2.4 데이터 파이프라인 베스트 프랙티스

- 원본 데이터를 스케일러블하고 증분적으로 인제스트
- raw 소스 데이터를 대상 테이블에 저장 (데이터 보존, 추적성, 감사)
- 강력한 데이터 클리닝 및 유효성 검사 프로세스 구현
- 데이터 볼륨 증가를 처리할 수 있도록 인제스트 파이프라인 설계
- 관련 메타데이터 추출 및 저장으로 검색 정확도 향상
- 로깅 및 모니터링으로 인제스트 성능 및 데이터 품질 추적

---

## 3. Layer 2: Query/Serving Pipeline (Online)

```
[Client] → [API Gateway/LB] → [FastAPI (async/await)] → [Retriever] → [Reranker] → [LLM] → [SSE Stream]
                                        ↕                      ↕
                                   [Response Cache]      [Vector DB]
```

### 3.1 사용자 대면 질의 — 동기 처리가 표준

사용자 대면 질의에 Queue를 넣는 것은 **일반적이지 않다**. 대부분의 프로덕션 RAG 서비스는 **동기 방식(FastAPI async/await)**으로 처리한다.

**이유:**
1. Queue를 거치면 큐잉/디큐잉 오버헤드로 레이턴시가 증가. 챗봇 UX에서 응답 지연은 치명적 (목표: 1.2~1.8초)
2. FastAPI의 async/await만으로도 높은 동시성 처리 가능
3. 수평 스케일링은 **로드밸런서 + 스테이트리스 워커**로 충분
4. **SSE(Server-Sent Events)**로 LLM 응답을 점진적으로 스트리밍하면 체감 응답 시간이 더 줄어듦

### 3.2 Queue가 적합한 경우

- 대량 배치 질의 (예: 1000건 문서 일괄 분류)
- 비동기 알림이 허용되는 경우 (이메일 보고서 생성 등)
- 극단적 트래픽 스파이크 흡수 (서킷 브레이커 역할)
- LLM 호출 비용 최적화를 위한 배치 처리

### 3.3 하이브리드 접근이 표준

실시간 사용자 질의는 동기 API, 백그라운드 작업(인제스션, 리인덱싱, 배치 분석)은 Queue.

### 3.4 프로덕션 성능 벤치마크

- 잘 최적화된 RAG 파이프라인: 평균 쿼리-응답 시간 1.2~1.8초
- 95th percentile 지연 시간: 복잡한 쿼리에서도 3초 이내
- 성숙한 RAG 파이프라인: 수십억 벡터에서도 쿼리 시간 100ms 이내

### 3.5 프로덕션 베스트 프랙티스

| 요소 | 설명 |
|------|------|
| **Multi-layer Caching** | 임베딩 캐시(1h TTL), 검색결과 캐시(30m), 시맨틱 응답 캐시. DoorDash는 이 방식으로 수십만 건 일일 처리, 2.5초 레이턴시 달성 |
| **Hybrid Search** | 벡터 검색(의미 유사) + 키워드 검색(정확 매칭) 결합. 검색 정확도 크게 향상 |
| **Reranker** | 1차 검색 결과를 Cross-encoder 등으로 재순위화. 관련성 높은 문서를 상위로 |
| **Model Routing** | 질의 복잡도에 따라 경량 모델(94.8%) / 고성능 모델(5.2%) 분배. 비용 25x 절감 가능 |
| **RAGOps (Observability)** | 검색 품질(precision@K), 생성 품질(faithfulness, hallucination rate), 시스템 성능(latency) 모니터링 |
| **Feedback Loop** | 사용자 피드백 수집 → 검색/생성 품질 지속 개선 |

---

## 4. Layer 3: Cross-Cutting Concerns

| 영역 | 상세 |
|------|------|
| **Observability (RAGOps)** | 모든 쿼리, 응답, 사용자 피드백, 컴포넌트 입출력 로깅. 포괄적 가시성 확보 |
| **Security** | 접근 제어, PII 마스킹. 프롬프트와 컨텍스트를 전용 큐로 라우팅하여 클라이언트 간 데이터 유출 방지. 민감 필드 자동 감지/삭제/해싱 후 임베딩 |
| **Feedback Loop** | 사용자 피드백으로 검색/생성 품질 지속 개선 |
| **Monitoring** | 검색 품질(context relevance, precision@K, hit rate), 생성 품질(answer relevancy, faithfulness, hallucination rate), 시스템 성능(latency, throughput, error rate) 추적 |

---

## 5. 심화 Q&A

### 5.1 Response Cache (Semantic Cache)

> **Q: Response Cache는 LLM 답변을 캐시하는 곳인가? 동일한 질의를 할 리가 없는데 의미가 있는가?**

핵심은 **Semantic Cache(의미 기반 캐시)**이다. 정확히 같은 문자열이 아니라 **의미적으로 유사한 질의**를 캐시 히트로 처리한다.

#### 작동 원리

```
사용자 질의 → 임베딩 변환 → 캐시 벡터 DB에서 유사도 검색
                                    ↓
                          유사도 > 임계값(예: 0.85)?
                           ├── Yes → 캐시된 응답 즉시 반환 (LLM 호출 생략)
                           └── No  → 전체 RAG 파이프라인 실행 → 결과를 캐시에 저장
```

#### 실제 예시

다음 질의들은 모두 **같은 캐시 히트**가 된다:
- "로그인이 안 돼요"
- "로그인 오류가 나요"
- "signin이 작동하지 않습니다"
- "로그인 할 수 없어요"

#### 실제 효과

- GPTCache 논문: 캐시 히트율 61~68%, API 호출 최대 68.8% 감소
- Portkey 프로덕션 데이터: Q&A 유스케이스에서 ~20% 캐시 히트율(99% 정확도)

#### 멀티레이어 캐시 운영

실제 프로덕션에서는 단일 캐시가 아니라 **3단계 캐시**를 운영한다:

| 레이어 | 대상 | TTL | 효과 |
|--------|------|-----|------|
| **Embedding Cache** | 같은 쿼리의 임베딩 재계산 방지 | ~1시간 | 임베딩 모델 호출 절감 |
| **Retrieval Cache** | 동일/유사 쿼리의 검색 결과 | ~30분 | Vector DB 부하 감소 |
| **Semantic Response Cache** | LLM 최종 응답 | ~15분~1시간 | LLM 비용 대폭 절감 |

#### 적합한 도메인 vs 부적합한 도메인

- **효과적**: 고객지원 봇, FAQ, 제품 Q&A, 사내 지식 검색 — 같은 주제의 변형 질의가 많음
- **비효과적**: 개인화된 분석 요청, 시시각각 변하는 데이터 기반 질의, 멀티턴 대화의 컨텍스트 의존적 질의

#### 도구

- [GPTCache](https://github.com/zilliztech/GPTCache) (오픈소스)
- PostgreSQL + pgvector로 직접 구현 가능

#### 캐시 관리 정책

- **TTL(Time-to-Live)**: 일정 시간 후 자동 만료
- **LRU(Least Recently Used)**: 가장 오래 사용되지 않은 항목부터 제거
- 고정 유사도 임계값(예: 0.8)은 모든 유스케이스에 일반화되지 않을 수 있음
- 도메인별 맞춤 임베딩이 필요할 수 있음
- 빠르게 변하는 데이터에는 TTL만으로 캐시 신선도 유지가 불충분

---

### 5.2 병목 해소 전략 — Scale Up vs Scale Out

> **Q: Vector DB 조회와 LLM 호출이 병목인데, Scale Up과 Scale Out 중 어떤 방향으로 해결하는가?**

결론: **둘 다 사용하되, 컴포넌트마다 주력 전략이 다르다.** Scale Up을 먼저 최대한 끌어올린 후, 그래도 부족하면 Scale Out하는 것이 비용 효율적이다.

```
┌─────────────────────────────────────────────┐
│  Scale Up (단일 인스턴스 최적화)              │
│  - Quantization (FP16→FP8)                  │
│  - PagedAttention + Continuous Batching     │
│  - 인덱싱 최적화 (IVF-PQ)                    │
│  ──── 여기까지 하고 나서 ────                 │
│                                             │
│  Scale Out (수평 확장)                       │
│  - LLM: 여러 vLLM 인스턴스 + LB             │
│  - Vector DB: 샤딩 + Read Replica           │
│  - Kubernetes 오토스케일링                    │
└─────────────────────────────────────────────┘
```

#### RAG 시스템의 주요 병목 포인트

1. **Database Encoding**: 컨텍스트 길이가 길어질수록(>1M) 인코딩 비용이 병목
2. **Retrieval Latency**: RAG 모델 크기를 늘려도 검색 성능이 제한 요인
3. **CPU-GPU 이종 파이프라인**: 외부 지식 검색(CPU) → LLM 실행(GPU)으로 병목이 GPU에서 CPU로 이동
4. **Vector Store Similarity Search**: 지식 베이스 커질수록 선형 스케일링하는 벡터 검색이 오버헤드
5. **디스크 데이터 전송**: 대규모 지식 베이스는 on-disk 저장 + 온디맨드 파티션 캐싱 → 데이터 전송 병목

#### Vector DB 병목 해소

| 전략 | 방법 | 적합한 상황 |
|------|------|------------|
| **Scale Out (주력)** | 샤딩 + Read Replica | 데이터량 증가, 높은 쿼리 동시성 |
| **Scale Up (보조)** | 인덱싱 최적화, GPU 가속 | 단일 노드 성능 극대화 |

**Scale Out 상세:**
- **샤딩**: 벡터 인덱스를 여러 노드에 분산. 질의가 각 샤드에 병렬 전송되어 top-k 결과를 머지. 검색 성능이 샤드 수에 거의 **선형 비례**하여 증가
  - 수평 샤딩: 메타데이터(날짜 범위, 카테고리) 기반으로 벡터를 노드 간 분산
  - LSH(Locality-Sensitive Hashing)로 유사 벡터를 같은 샤드에 그룹핑, 교차 노드 검색 감소
  - Pinecone: 계층적 그래프 인덱싱으로 교차 노드 레이턴시 완화
  - 샤드 간 복제로 노드 장애 시에도 데이터 가용성 보장 (Elasticsearch)
- **Read Replica**: 읽기 전용 복제본으로 읽기 부하 분산. RAG는 read-heavy 워크로드에 특히 적합
- 글로벌 애플리케이션: 여러 지리적 리전에 배포하여 지연 감소

**Scale Up 상세:**
- HNSW → IVF-PQ 인덱스 전환으로 검색 범위 축소 (📖 [HNSW vs IVF-PQ 심층 비교](./hnsw-vs-ivfpq-deep-dive.md))
- GPU 가속 벡터 검색 (NVIDIA cuVS)
- 인메모리 캐싱 (Redis)으로 hot 데이터 빠른 접근
- 임베딩 캐싱: 10K개 768-d 벡터(FP16)를 캐싱해도 15MB CPU 메모리만 필요

#### LLM 병목 해소

| 전략 | 방법 | 적합한 상황 |
|------|------|------------|
| **Scale Up (단일 GPU 최적화)** | Quantization, PagedAttention | 비용 최적화, 지연 시간 단축 |
| **Scale Out (복제)** | Data Parallelism, 멀티 인스턴스 | 높은 동시 요청 처리 |

**Scale Up 상세:**
- **Quantization**: FP16 → FP8/INT4로 정밀도 낮춰 메모리 절감 + 속도 2배 향상
  - NVIDIA NeMo: TensorRT Model Optimizer로 PTQ(Post-Training Quantization) 적용
  - "FP8 대비 FP16에서 최대 2배 속도 향상"
- **PagedAttention** (vLLM): KV 캐시를 비연속적 블록으로 관리하여 GPU 메모리 낭비 제거 → 배치 크기 증가
- **Continuous Batching**: 완료된 시퀀스를 즉시 새 요청으로 교체하여 GPU 유휴 시간 제거
- 70B 파라미터 모델: FP16에서 약 140GB 메모리 필요 → 단일 GPU 불가 → 양자화 필수

**Scale Out 상세:**
- **Data Parallelism**: 동일 모델을 여러 GPU에 복제, 로드밸런서로 요청 분배. 처리량이 인스턴스 수에 선형 비례
- 큰 모델(70B+): **Tensor Parallelism**(GPU 간 행렬 연산 분할) + **Pipeline Parallelism**(레이어별 분할) 조합
  - 예: 8 GPU 노드 2개 → `tensor_parallel_size=8`, `pipeline_parallel_size=2`
- Kubernetes 오토스케일링: GPU 사용률 기반으로 인스턴스 자동 증감
- 비-MoE 모델: 여러 독립 vLLM 인스턴스를 로드밸런싱하는 것도 가능

**vLLM vs TGI:**
- vLLM: 높은 동시성에서 **최대 24배 높은 처리량** (PagedAttention). 고처리량 배치 처리에 적합
- TGI: 단일 사용자 인터랙티브 시나리오에서 더 낮은 tail latency. 지연 민감 애플리케이션에 적합

#### 주요 프레임워크/도구 (2025)

| 도구/논문 | 초점 |
|---|---|
| **RAGO** (ISCA 2025) | 체계적 RAG 서빙 성능 최적화. 베이스라인 대비 1.7배 QPS/Chip 향상 |
| **RAG-Stack** | Vector DB 관점에서 RAG 품질과 성능 동시 최적화 |
| **HERMES** | 이종 하드웨어에서의 다단계 AI 추론 파이프라인 시뮬레이션 |
| **RAGDoll** | 단일 GPU에서 효율적 오프로딩 기반 RAG |
| **NVIDIA RAG Blueprint** | NIM 마이크로서비스 + GPU 가속 벡터 검색(cuVS) 참조 아키텍처 |

---

### 5.3 SSE Streaming 동작 방식

> **Q: LLM이 처리하면 SSE Stream으로 응답하는데, FastAPI는 작업을 받았다는 정보를 이미 return한 건가?**

SSE는 **Queue 기반 비동기(202 Accepted)와 완전히 다른 방식**이다. "작업 받았다는 정보를 return"하는 게 아니라, HTTP 200과 함께 **응답 헤더를 먼저 보내고, 연결을 끊지 않는다.**

#### SSE의 실제 동작 흐름

```
시간 →

Client                    FastAPI                    LLM
  │                         │                         │
  │── POST /ask ──────────→│                         │
  │                         │── retrieve + rerank ──→│
  │                         │                         │
  │←─ HTTP 200 ────────────│  (Content-Type:         │
  │   (연결 유지!)           │   text/event-stream)   │
  │                         │                         │
  │←─ data: "안녕" ─────────│←── token: "안녕" ──────│
  │←─ data: "하세요" ───────│←── token: "하세요" ────│
  │←─ data: "," ────────────│←── token: "," ─────────│
  │←─ data: "저는" ─────────│←── token: "저는" ──────│
  │←─ data: "..." ──────────│←── token: "..." ───────│
  │←─ data: [DONE] ─────────│                         │
  │                         │                         │
  │── (연결 종료) ──────────│                         │
```

#### 핵심 포인트

1. FastAPI가 HTTP 200과 `Content-Type: text/event-stream` 헤더를 먼저 보내고 **연결을 유지**
2. 같은 HTTP 연결 위에서 LLM이 토큰을 생성할 때마다 `data: {토큰}\n\n` 형태로 점진적 전송
3. 클라이언트는 첫 토큰부터 화면에 표시 → **체감 응답 시간이 훨씬 짧음**
4. 모든 토큰 전송 후 연결 종료
5. SSE는 **일반 HTTP 위에서 동작** → CDN, 리버스 프록시, 방화벽과 호환성 좋음
6. 클라이언트 **자동 재연결** 내장

#### FastAPI 코드 예시

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

async def generate_stream(query: str):
    """LLM 토큰을 하나씩 yield하는 async generator"""
    # 1. Retrieve (동기적으로 완료)
    results = await retriever.retrieve(query)
    context = format_context(results)

    # 2. LLM Streaming - 토큰 단위로 yield
    async for token in llm.stream(query, context):
        yield f"data: {token}\n\n"     # SSE 프로토콜 형식

    yield "data: [DONE]\n\n"           # 스트림 종료 신호

@app.post("/ask")
async def ask(query: str):
    return StreamingResponse(
        generate_stream(query),
        media_type="text/event-stream",  # ← 이 헤더가 SSE를 활성화
    )
```

#### 3가지 응답 패턴 비교

| 패턴 | 흐름 | 용도 |
|------|------|------|
| **동기 (일반 REST)** | `POST → 전체 응답 완성 후 200 + body 반환` | 짧은 응답, 단순 API |
| **비동기 Queue** | `POST → 202 Accepted + job_id → 별도 GET으로 결과 폴링` | 배치 처리, 오래 걸리는 작업 |
| **SSE Streaming** | `POST → 200 + 헤더 즉시 반환 → 같은 연결에서 토큰 점진 전송 → 종료` | **LLM 채팅, RAG Q&A** |

ChatGPT, Claude 웹 인터페이스 모두 이 SSE 패턴을 사용한다.

#### SSE vs WebSocket

- SSE: 단방향(서버→클라이언트), 일반 HTTP, CDN/프록시 친화적, 자동 재연결
- WebSocket: 양방향, 연결 업그레이드 필요, 로드밸런서 설정 복잡, sticky session 필요
- LLM 토큰 스트리밍은 서버→클라이언트 단방향이므로 **SSE가 적합**

#### 수정된 아키텍처 다이어그램

```
[Client]
   │
   │ POST /ask (query)
   ▼
[API Gateway / LB]
   │
   ▼
[FastAPI (async)]
   │
   ├── 1. Cache Check ──→ [Semantic Cache] ──→ 히트 시 즉시 반환
   │
   ├── 2. Retrieve ──→ [Vector DB] (Hybrid Search)
   │
   ├── 3. Rerank ──→ [Reranker Model]
   │
   └── 4. Generate ──→ [LLM (vLLM/TGI)]
         │
         │ (토큰 단위 스트리밍)
         ▼
   ←── SSE: data: token1\n\n
   ←── SSE: data: token2\n\n
   ←── SSE: data: token3\n\n
   ←── SSE: data: [DONE]\n\n
   │
   └── (연결 종료)
```

Step 1~3(캐시 확인, 검색, 리랭킹)은 **동기적으로 완료** 후, Step 4(LLM 생성)부터 **스트리밍이 시작**된다. 클라이언트는 검색/리랭킹이 끝난 직후부터 첫 토큰을 받기 시작한다.

---

## 6. 현재 코드베이스와의 갭 분석

### 6.1 모듈별 프로덕션 전환 필요사항

| 모듈 | 현재 구현 | 프로덕션 필요 사항 |
|------|----------|-----------------|
| `generator.py` | `subprocess.run(["claude", "-p", ...])` CLI 호출 | Anthropic API SDK (async, streaming, 토큰 카운팅) |
| `embedder.py` | 글로벌 Singleton, thread-unsafe | Connection pool 또는 async 모델 서빙 (Ray Serve, BentoML) |
| `store.py` | 매 호출마다 `VectorStore()` 새로 생성 | 커넥션 풀링, 영구 인스턴스 유지 |
| `pipeline.py` | 동기 함수, stateless | Ingestion/Serving 서비스 분리, async 지원 |
| `splitter.py` | 문자 단위 청킹 | Semantic chunking (문장 경계 인식) |
| `loader.py` | `.txt`만 지원 | PDF, DOCX, HTML 등 멀티포맷 |
| `cli.py` | Typer CLI | FastAPI 서버 + API Gateway |
| ChromaDB (로컬) | 단일 프로세스 | Managed Vector DB (Pinecone/Milvus/Weaviate) |
| 버전 관리 없음 | — | Metadata Registry + Document Versioning |
| 캐싱 없음 | — | Multi-layer Cache (Redis 등) |
| 모니터링 없음 | — | RAGOps 인프라 |

### 6.2 우선순위

**가장 급한 3가지:**
1. CLI → API 전환 (generator.py의 subprocess 호출을 Anthropic API SDK로)
2. Ingestion/Serving 서비스 분리
3. async/await 도입

이 3가지가 해결되면 나머지는 점진적으로 추가할 수 있다.

### 6.3 아키텍처 갭

1. **임베딩 모델 추상화 없음**: 하드코딩된 `all-MiniLM-L6-v2`. 모델 교체를 위한 인터페이스 필요
2. **LLM 프로바이더 추상화 없음**: 하드코딩된 Claude CLI. OpenAI, Anthropic API, 로컬 모델 등을 위한 인터페이스 필요
3. **벡터 스토어 추상화 없음**: 하드코딩된 ChromaDB. Pinecone, Weaviate 등을 위한 인터페이스 필요
4. **설정 관리 없음**: 모듈 곳곳에 하드코딩된 기본값. 중앙 집중식 설정 파일/환경 변수 필요
5. **의존성 주입 없음**: 수동 컴포넌트 초기화. DI 컨테이너로 테스트/교체 용이성 확보 필요

---

## 7. Sources

### 프로덕션 RAG 아키텍처
- [Building a Scalable, Production-Grade Agentic RAG Pipeline](https://levelup.gitconnected.com/building-a-scalable-production-grade-agentic-rag-pipeline-1168dcd36260)
- [RAG Architecture Explained: A Comprehensive Guide](https://orq.ai/blog/rag-architecture)
- [The Architect's Guide to Production RAG](https://www.ragie.ai/blog/the-architects-guide-to-production-rag-navigating-challenges-and-building-scalable-ai)
- [RAGOps: Operating and Managing RAG Pipelines (arXiv)](https://arxiv.org/html/2506.03401v1)
- [Bringing Your RAG System to Life - The Data Pipeline](https://jamwithai.substack.com/p/bringing-your-rag-system-to-life)
- [RAG in Production: Deployment Strategies](https://coralogix.com/ai-blog/rag-in-production-deployment-strategies-and-practical-considerations/)
- [8 RAG Patterns You Should Stop Ignoring](https://dev.to/neurondb_support_d73fa7ba/retrieval-augmented-generation-architectures-patterns-and-production-reality-49g1)

### Data Ingestion & Versioning
- [Optimize Your RAG Pipeline with Proper Data Ingestion](https://www.pryon.com/resource/5-things-to-consider-when-building-your-own-rag-ingestion-pipeline)
- [How to Update RAG Knowledge Base Without Rebuilding](https://particula.tech/blog/update-rag-knowledge-without-rebuilding)
- [The Knowledge Decay Problem in RAG Systems](https://ragaboutit.com/the-knowledge-decay-problem-how-to-build-rag-systems-that-stay-fresh-at-scale/)
- [Databricks - Build Unstructured Data Pipeline for RAG](https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag)
- [RAG Data Ingestion - AI Engineering Academy](https://aiengineering.academy/RAG/01_Data_Ingestion/data_ingestion/)

### Semantic Cache
- [Semantic Cache: How to Speed Up LLM and RAG Applications](https://medium.com/@svosh2/semantic-cache-how-to-speed-up-llm-and-rag-applications-79e74ce34d1d)
- [GPT Semantic Cache (arXiv)](https://arxiv.org/html/2411.05276v2)
- [Portkey - Semantic Cache for LLMs](https://portkey.ai/blog/reducing-llm-costs-and-latency-semantic-cache/)
- [GPTCache - GitHub](https://github.com/zilliztech/GPTCache)
- [How to Reduce Cost and Latency Using Semantic LLM Caching](https://www.marktechpost.com/2025/11/11/how-to-reduce-cost-and-latency-of-your-rag-application-using-semantic-llm-caching/)

### Scaling & Performance
- [RAGO: Systematic RAG Performance Optimization (ISCA 2025)](https://people.csail.mit.edu/suvinay/pubs/2025.rago.isca.pdf)
- [NVIDIA - Horizontal Autoscaling of RAG on Kubernetes](https://developer.nvidia.com/blog/enabling-horizontal-autoscaling-of-enterprise-rag-components-on-kubernetes)
- [NVIDIA - RAG on GH200](https://developer.nvidia.com/blog/deploying-retrieval-augmented-generation-applications-on-nvidia-gh200-delivers-accelerated-performance/)
- [vLLM Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [Scaling RAG to 20M Docs](https://www.chitika.com/scaling-rag-20-million-documents/)
- [Scale Vector Search: Sharding & Replication](https://apxml.com/courses/large-scale-distributed-rag/chapter-2-advanced-distributed-retrieval-strategies/scaling-vector-search-sharding-replication)
- [Milvus - Batch Processing in RAG Systems](https://milvus.io/ai-quick-reference/how-do-batch-processing-or-asynchronous-calls-improve-the-throughput-of-a-rag-system-and-what-is-the-effect-on-singlequery-latency)
- [Async & Batching in RAG](https://apxml.com/courses/optimizing-rag-for-production/chapter-4-end-to-end-rag-performance/async-processing-batching-rag)
- [Ray Serve: Async Inference](https://www.anyscale.com/blog/ray-serve-autoscaling-async-inference-custom-routing)
- [RAG Performance Optimization with TensorRT](https://www.codespace.blog/performance-optimization-with-nvidia-tensorrt-and-quantization/)
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm)

### SSE Streaming
- [FastAPI + SSE for LLM Tokens](https://medium.com/@hadiyolworld007/fastapi-sse-for-llm-tokens-smooth-streaming-without-websockets-001ead4b5e53)
- [SSE with FastAPI and React (LangGraph)](https://www.softgrade.org/sse-with-fastapi-react-langgraph/)
- [Streaming AI Agent Responses with SSE](https://akanuragkumar.medium.com/streaming-ai-agents-responses-with-server-sent-events-sse-a-technical-case-study-f3ac855d0755)
- [Implementing SSE with FastAPI](https://mahdijafaridev.medium.com/implementing-server-sent-events-sse-with-fastapi-real-time-updates-made-simple-6492f8bfc154)
- [Streaming Responses in FastAPI](https://hassaanbinaslam.github.io/posts/2025-01-19-streaming-responses-fastapi.html)

### Vector Database 비교
- [Best Vector Databases for RAG: 2025 Comparison Guide](https://latenode.com/blog/ai-frameworks-technical-infrastructure/vector-databases-embeddings/best-vector-databases-for-rag-complete-2025-comparison-guide)
- [We Tried 10 Best Vector Databases for RAG](https://www.zenml.io/blog/vector-databases-for-rag)
- [Production-Ready RAG: Engineering Guidelines](https://www.netguru.com/blog/rag-for-scalable-systems)
