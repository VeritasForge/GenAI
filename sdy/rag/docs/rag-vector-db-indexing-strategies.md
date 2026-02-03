# RAG Vector DB 인덱싱 전략

## 1. HNSW vs IVF-PQ: 문서 RAG에서의 선택

일반적인 문서 기반 RAG에서는 **HNSW가 더 적합한 선택**이다.

### HNSW가 문서 RAG에 적합한 이유

1. **높은 recall**: HNSW는 파라미터 튜닝(`ef_search`)에 따라 99%+ recall 달성 가능. RAG에서는 관련 문서를 놓치면 답변 품질이 직접 저하되므로 recall이 핵심
2. **빠른 검색 속도**: 그래프 기반 탐색으로 로그 스케일 검색
3. **규모 적합성**: 대부분의 기업 문서 RAG는 수십만~수백만 청크 수준이고, 이 규모에서 HNSW는 메모리에 충분히 올라감

### IVF-PQ의 트레이드오프

문서 295행에서 "HNSW → IVF-PQ 인덱스 전환으로 검색 범위 축소"라고 한 것은 **Scale Up 전략** 맥락이다. IVF-PQ는:

- **PQ(Product Quantization)**가 벡터를 손실 압축하므로 recall이 떨어짐
- `nprobe` 파라미터로 recall/속도 트레이드오프를 조절할 수 있지만, 같은 속도에서 HNSW보다 recall이 낮음
- 대신 메모리 사용량을 10~50배 줄일 수 있음

### IVF-PQ가 필요한 시점

벡터가 **수억~수십억 개** 수준으로 커져서 HNSW로는 메모리가 감당이 안 될 때이다. 이때는 recall을 일부 희생하더라도 IVF-PQ(또는 HNSW + PQ 하이브리드)로 전환하거나, 샤딩으로 분산하는 것이 현실적인 선택이 된다.

결론적으로, 일반적인 문서 RAG 규모에서는 **정확도(recall)가 우선**이므로 HNSW가 맞고, IVF-PQ는 스케일 문제가 발생했을 때 고려하는 옵션이다.

---

## 2. Vector DB Write/Read 분리 (Replica)

Vector DB도 일반적인 DB처럼 **Write/Read 분리** 구조를 적용할 수 있다.

### RAG에서의 Write/Read 분리

문서 291행에서도 언급하고 있듯이:

> "Read Replica: 읽기 전용 복제본으로 읽기 부하 분산. RAG는 read-heavy 워크로드에 특히 적합"

RAG는 본질적으로 **Ingestion(Write)은 가끔, Query(Read)는 빈번**한 read-heavy 워크로드이므로 Write/Read 분리가 특히 효과적이다.

### 구조

```
[Ingestion Pipeline (Offline)]
        │
        ▼ Write
   [Primary Node (HNSW Index)]
        │
        ├── Replicate ──→ [Read Replica 1] ←── Query
        ├── Replicate ──→ [Read Replica 2] ←── Query
        └── Replicate ──→ [Read Replica N] ←── Query
                              ▲
                         [Load Balancer]
                              ▲
                      [Serving Pipeline]
```

### 주요 Vector DB별 지원 현황

| Vector DB | Write/Read 분리 | 방식 |
|-----------|----------------|------|
| **Milvus** | O | Read Replica + 샤딩 기본 지원 |
| **Weaviate** | O | Raft 기반 복제, read consistency level 설정 가능 |
| **Qdrant** | O | 분산 모드에서 복제본 설정 |
| **Pinecone** | O (관리형) | 내부적으로 자동 처리 |
| **ChromaDB** | X (현재) | 단일 프로세스, 프로덕션용이 아님 |

### 고려할 점

- **Replication Lag**: Write 후 Read Replica에 반영되기까지 약간의 지연이 있음. 문서 인덱싱은 실시간이 아니라 배치로 하는 경우가 많으므로 대부분 문제 없음
- **Consistency Level**: 강한 일관성(strong) vs 최종 일관성(eventual) 선택 가능. RAG에서는 문서가 몇 초 늦게 반영되어도 큰 문제가 없으므로 eventual consistency로 충분
- **HNSW 인덱스 동기화**: HNSW 그래프 자체가 복제되어야 하므로, 새 문서 추가 시 각 replica에서 인덱스 업데이트가 필요. 이 비용이 있긴 하지만 read 성능 향상으로 상쇄됨

현재 프로젝트의 ChromaDB는 이 구조를 지원하지 않으므로, 프로덕션 전환 시 문서 467행에서 언급한 대로 Milvus나 Weaviate 같은 분산 지원 Vector DB로 전환이 필요하다.

---

## 3. Vector DB 메타데이터 인덱싱

Vector DB도 메타데이터 필터링을 위한 인덱싱을 지원한다. 다만 RDB와는 방식이 다르다.

### Vector DB의 메타데이터 필터링 방식

Vector 검색과 메타데이터 필터링을 결합하는 전략은 크게 3가지이다:

```
방식 1: Pre-filtering (필터 먼저)
[메타데이터 필터] → 후보 축소 → [HNSW 검색]
  장점: 정확한 필터링
  단점: 후보가 적으면 HNSW 그래프가 sparse해져 recall 저하

방식 2: Post-filtering (검색 먼저)
[HNSW 검색] → top-K 결과 → [메타데이터 필터]
  장점: 검색 품질 유지
  단점: 필터 후 결과가 K보다 훨씬 적을 수 있음

방식 3: In-filter (검색 중 필터) ← 대부분의 프로덕션 DB가 채택
[HNSW 탐색하면서 동시에 메타데이터 조건 체크]
  장점: recall과 필터 정확도 모두 유지
  단점: 구현 복잡
```

### 주요 Vector DB의 메타데이터 인덱싱

| Vector DB | 메타데이터 인덱스 | 방식 |
|-----------|----------------|------|
| **Milvus** | 명시적 인덱스 생성 가능 | `create_index(field="author")` — Inverted Index, Trie 등 선택 |
| **Weaviate** | 자동 인덱싱 | Roaring Bitmap 기반, filterable 속성에 자동 적용 |
| **Qdrant** | 명시적 인덱스 생성 | `create_payload_index(field="category", schema="keyword")` |
| **Pinecone** | 자동 처리 | 내부적으로 메타데이터 인덱싱 |
| **ChromaDB** | 제한적 | 단순 필터링은 되지만 별도 인덱스 최적화 없음 |

### RDB 인덱스와의 비교

```
RDB:
  CREATE INDEX idx_author ON documents(author);
  SELECT * FROM documents WHERE author = 'Kim' ORDER BY score;

Qdrant (예시):
  # 메타데이터 필드에 인덱스 생성
  client.create_payload_index(
      collection_name="documents",
      field_name="author",
      field_schema=models.PayloadSchemaType.KEYWORD
  )

  # 벡터 검색 + 메타데이터 필터 결합
  client.search(
      collection_name="documents",
      query_vector=query_embedding,
      query_filter=Filter(
          must=[FieldCondition(key="author", match=MatchValue(value="Kim"))]
      ),
      limit=10
  )

Milvus (예시):
  # 스칼라 필드에 인덱스 생성
  collection.create_index(
      field_name="category",
      index_type="INVERTED"
  )

  # 벡터 검색 + 필터
  collection.search(
      data=[query_embedding],
      anns_field="embedding",
      param={"metric_type": "COSINE"},
      filter='category == "finance" and year >= 2024',
      limit=10
  )
```

### 핵심 차이점

| | RDB | Vector DB |
|---|---|---|
| **주 검색** | 정확 매칭 (WHERE) | 유사도 검색 (ANN) |
| **인덱스 대상** | 스칼라 컬럼 (B-Tree, Hash) | 벡터 (HNSW, IVF) + 스칼라 메타데이터 |
| **필터 역할** | 주요 조회 조건 | 벡터 검색의 보조 조건 |
| **인덱스 타입** | B-Tree, Hash, GIN 등 | 메타데이터용: Inverted Index, Bitmap, Trie |

결론적으로 Vector DB도 메타데이터 필드에 인덱스를 걸 수 있고, 프로덕션에서는 **반드시 걸어야** 한다. 메타데이터 필터 없이 전체 벡터를 검색하는 것과, `category = "finance"`로 범위를 좁힌 후 검색하는 것은 성능 차이가 크다. 특히 NLP/NER로 추출한 메타데이터(author, date, category 등)를 필터링에 활용하려면 해당 필드에 인덱스가 있어야 효율적이다.

---

## 4. 메타데이터 인덱스 선정 기준

모든 메타데이터 필드에 인덱스를 거는 것은 비효율적이고, RDB와 마찬가지로 **선택적으로** 걸어야 한다.

### 인덱스를 걸어야 하는 기준

**RDB와 동일한 원칙**이 적용된다 — **쿼리 패턴(어떤 필터를 자주 쓰는가)**이 기준이다.

```
NLP/NER로 추출 가능한 메타데이터 예시:

  author        ← 자주 필터링?
  date          ← 자주 필터링?
  category      ← 자주 필터링?
  department    ← 자주 필터링?
  language      ← 가끔?
  entities[]    ← 거의 안 씀?
  keywords[]    ← 거의 안 씀?
  sentiment     ← 거의 안 씀?
  source_url    ← 거의 안 씀?
```

### 판단 기준

| 기준 | 인덱스 O | 인덱스 X |
|------|---------|---------|
| **쿼리 빈도** | 대부분의 쿼리에서 필터로 사용 | 거의 필터링에 안 쓰임 |
| **카디널리티** | 적절한 카디널리티 (수십~수천) | 너무 높음(unique값) 또는 너무 낮음(2~3개) |
| **필터 선택도** | 데이터를 의미 있게 줄여줌 | 필터링해도 거의 안 줄어듦 |
| **용도** | 검색 조건 | 단순 표시/저장용 |

### 실제 문서 RAG에서의 예시

```
✅ 인덱스 추천 (자주 필터링, 선택도 높음)
─────────────────────────────
category      : "finance", "hr", "legal" 등 — 거의 매 쿼리에서 사용
date/year     : 범위 검색 (year >= 2024) — 최신 문서 우선 검색
department    : 부서별 문서 접근 제어에도 활용
document_type : "policy", "report", "manual" 등
access_level  : 권한 기반 필터링 (보안)

⚠️ 상황에 따라 (쿼리 패턴 확인 후 결정)
─────────────────────────────
author        : 특정 저자 검색이 빈번하면 O
language      : 다국어 환경이면 O, 단일 언어면 X
tags/labels   : 태그 기반 검색을 UI에서 지원하면 O

❌ 인덱스 불필요 (저장/표시용, 필터에 안 쓰임)
─────────────────────────────
source_url    : 결과 표시용, 필터 조건 아님
chunk_index   : 내부 관리용
raw_entities  : NER 결과 전체 리스트 — 너무 세분화
sentiment     : 일반 문서 RAG에서 감성으로 필터링할 일 거의 없음
summary       : 텍스트 필드, 필터 대상 아님
```

### 인덱스 과다의 비용

RDB와 마찬가지로 인덱스가 많으면 비용이 발생한다:

| 비용 | 설명 |
|------|------|
| **Write 성능 저하** | 문서 추가/업데이트 시 모든 인덱스를 갱신해야 함 |
| **메모리 사용 증가** | 각 인덱스가 메모리를 점유 |
| **인덱스 빌드 시간** | 대량 인덱싱 시 오버헤드 |

### 실무 접근법

```
1단계: 인덱스 없이 시작
       → 메타데이터는 저장하되, 인덱스는 안 걸음

2단계: 쿼리 로그 분석
       → 실제로 어떤 필터가 자주 쓰이는지 확인
       → slow query 확인

3단계: 상위 빈도 필터 필드에만 인덱스 추가
       → 보통 3~5개 필드면 충분

4단계: 모니터링
       → 필터링 성능 추적, 필요 시 추가/제거
```

결국 RDB의 인덱스 전략과 같다. **"어떤 컬럼에 인덱스를 걸지"가 아니라 "사용자가 어떤 조건으로 검색하는지"**가 기준이고, NLP/NER로 추출했다고 전부 인덱싱하는 것이 아니라 실제 쿼리 패턴에 따라 선별적으로 적용한다.
