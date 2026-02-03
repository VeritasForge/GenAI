# RAG 시스템 캐싱 전략 심층 분석

> 기존 `production-rag-architecture.md`의 3-Layer 캐시 모델(Embedding Cache -> Retrieval Cache -> Response Cache)에 오류가 있어, 권위 있는 출처를 기반으로 새로 조사한 내용입니다.

---

## 1. 기존 모델의 문제점

### 1.1 3-Layer 캐시 모델이 잘못된 이유

기존 문서에서 제시한 모델:

```
Embedding Cache (TTL ~1hr) → Retrieval Cache (TTL ~30min) → Response Cache (TTL ~15min-1hr)
```

**문제 1: Embedding Cache는 별도 레이어로 존재하지 않는다**

- 대부분의 권위 있는 출처(HuggingFace, GPTCache, Redis, Qdrant)에서 "Embedding Cache"를 별도 레이어로 분류하지 않음
- **닭과 달걀 문제**: 시맨틱 캐시를 하려면 임베딩이 필요한데, 임베딩 캐시의 목적이 임베딩 생성을 건너뛰는 것이므로 모순
- Exact match 기반 Embedding Cache는 "로그인 안됨" vs "로그인이 안되고 있어" 같은 동의어 쿼리를 구분하지 못해 캐시 히트율이 ~15%에 불과

**문제 2: 실제 업계 표준은 2-Approach 모델**

권위 있는 출처들이 공통적으로 제시하는 구조:

```
┌─────────────────────────────────────────────────────────────────┐
│                RAG 캐시: 2-Approach 모델                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Approach A: Pre-Retrieval Cache                                │
│  ┌──────┐    ┌──────────┐    ┌──────────┐    ┌─────┐           │
│  │ User │───▶│ Embedding │───▶│ Cache    │───▶│ LLM │           │
│  └──────┘    └──────────┘    │ (문서 청크)│    └─────┘           │
│                              └─────┬────┘                       │
│                                miss│                            │
│                                    ▼                            │
│                              ┌──────────┐                       │
│                              │ Vector DB│                       │
│                              └──────────┘                       │
│                                                                 │
│  Approach B: Post-Retrieval Cache                               │
│  ┌──────┐    ┌──────────┐    ┌──────────┐                      │
│  │ User │───▶│ Embedding │───▶│ Cache    │──hit──▶ 응답 반환     │
│  └──────┘    └──────────┘    │(LLM 응답) │                      │
│                              └─────┬────┘                       │
│                                miss│                            │
│                                    ▼                            │
│                              ┌──────────┐    ┌─────┐           │
│                              │ Vector DB│───▶│ LLM │           │
│                              └──────────┘    └─────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Semantic Cache의 기본 동작 원리

### 2.1 시맨틱 캐시란?

쿼리의 **의미적 유사도**를 기반으로 캐시 히트를 판단하는 방식이다. 정확한 문자열 일치(exact match)가 아니라, 임베딩 벡터 간 거리를 계산하여 유사한 쿼리를 식별한다.

```
┌────────────────────────────────────────────────────────────────┐
│                    Semantic Cache 동작 흐름                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. 쿼리 임베딩 생성 (항상 실행)                                  │
│     "로그인이 안돼요" → [0.12, -0.45, 0.78, ...]               │
│                                                                │
│  2. 캐시 인덱스에서 유사 쿼리 검색 (FAISS, etc.)                  │
│     기존 캐시: "로그인 안됨" → [0.11, -0.44, 0.79, ...]        │
│     거리 계산: 0.023 (매우 가까움)                               │
│                                                                │
│  3. 임계값 비교                                                  │
│     0.023 < threshold(0.35) → 캐시 히트!                       │
│                                                                │
│  4. 캐시된 결과 반환                                             │
│     (문서 청크 또는 LLM 응답 — 캐시 유형에 따라 다름)              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 핵심: 임베딩 생성은 항상 필요하다

두 가지 접근법 모두 **쿼리 임베딩 생성을 건너뛸 수 없다**. 시맨틱 유사도를 판단하려면 현재 쿼리의 임베딩이 필요하기 때문이다.

```
                    ┌──────────────────┐
                    │  사용자 쿼리 입력   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  임베딩 생성       │  ← 이 단계는 절대 건너뛸 수 없음
                    │  (항상 실행)       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  캐시 조회         │
                    │  (유사도 검색)     │
                    └────────┬─────────┘
                             │
                    hit ─────┼───── miss
                    │                 │
                    ▼                 ▼
              캐시 결과 반환      전체 파이프라인 실행
```

### 2.3 유사도 임계값 설정 가이드

| 임계값 | 캐시 히트율 | 오류율 | 적합한 도메인 |
|:------:|:----------:|:------:|-------------|
| 0.80 | ~68% | ~3% | 일반 FAQ, 고객센터 |
| 0.85 | ~55% | ~1.5% | 일반 비즈니스 |
| 0.90 | ~40% | <1% | 기술 문서 |
| 0.95 | ~20% | <0.5% | 금융, 의료, 법률 |

---

## 3. Pre-Retrieval Cache (Approach A)

### 3.1 개요

**캐시 위치**: 사용자 쿼리 → [Cache] → Vector DB
**캐시 대상**: Vector DB 검색 결과 (문서 청크)
**LLM 호출**: 매번 실행

### 3.2 동작 방식

HuggingFace Cookbook의 구현을 기반으로 한 동작 흐름:

```python
class semantic_cache:
    def __init__(self, thresold=0.35):
        self.index = faiss.IndexFlatL2(768)  # FAISS 인덱스
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        self.cache = {
            'questions': [],      # 이전 쿼리 텍스트
            'embeddings': [],     # 이전 쿼리 임베딩
            'answers': [],        # Vector DB 검색 결과 (문서 청크!)
            'response_text': []   # 검색 결과 텍스트
        }

    def ask(self, question):
        # 1. 항상 임베딩 생성
        embedding = self.encoder.encode([question])

        # 2. FAISS에서 유사 쿼리 검색
        D, I = self.index.search(embedding, 1)

        if D[0][0] < self.euclidean_threshold:
            # 3a. 캐시 히트 → 이전 검색 결과(문서 청크) 반환
            return self.cache['response_text'][I[0][0]]
        else:
            # 3b. 캐시 미스 → Vector DB 검색
            result = query_database(question)
            # 캐시에 저장
            self.cache['questions'].append(question)
            self.cache['embeddings'].append(embedding.tolist())
            self.cache['response_text'].append(result)
            self.index.add(embedding)
            return result
```

### 3.3 Pre-Retrieval의 핵심 가치

HuggingFace Cookbook 저자의 명시적 설명:

> "Placing it at the model's response point may lead to a **loss of influence over the obtained response**. The cache system could consider 'Explain the French Revolution in 10 words' and 'Explain the French Revolution in a hundred words' as the same query."

> "While the model's response may differ based on the request, **the information retrieved from the vector database should be the same**."

```
┌────────────────────────────────────────────────────────────────────┐
│                  Pre-Retrieval Cache의 핵심 장점                     │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  [쿼리 A] "프랑스 혁명을 10단어로 설명해"                              │
│     → 임베딩 → 캐시 미스 → Vector DB 검색                            │
│     → 검색 결과(문서 청크 3개) 캐시에 저장                             │
│     → 문서 청크 + "10단어로" 지시 → LLM → 짧은 응답 ✅                │
│                                                                    │
│  [쿼리 B] "프랑스 혁명을 100단어로 상세히 설명해"                       │
│     → 임베딩 → 캐시 히트! (같은 문서 청크 반환)                        │
│     → 문서 청크 + "100단어로 상세히" 지시 → LLM → 상세 응답 ✅         │
│                                                                    │
│  ✅ 검색 비용 절약 + 사용자 지시 정확히 반영                            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 3.4 HuggingFace Cookbook 실행 결과

| 쿼리 | 결과 | 거리 | 소요 시간 |
|------|------|:----:|:---------:|
| "How do vaccines work?" | ChromaDB 검색 (미스) | - | 0.057s |
| "Briefly explain me what is a Sydenham chorea." | 캐시 히트 | 0.028 | 0.019s |
| "Write in 20 words what is a Sydenham chorea." | 캐시 히트 | 0.228 | 0.016s |

---

## 4. Post-Retrieval Cache (Approach B)

### 4.1 개요

**캐시 위치**: 사용자 쿼리 → [Cache] → (Vector DB + LLM)
**캐시 대상**: LLM 최종 응답
**LLM 호출**: 캐시 히트 시 건너뜀

### 4.2 동작 방식

대표 구현: [GPTCache](https://github.com/zilliztech/GPTCache)

```
사용자 쿼리
    │
    ▼
임베딩 생성
    │
    ▼
캐시 유사도 검색 ──hit──▶ 캐시된 LLM 응답 반환 (Vector DB + LLM 모두 건너뜀)
    │
   miss
    │
    ▼
Vector DB 검색 → LLM 호출 → 응답 생성
    │
    ▼
캐시에 저장 (쿼리 임베딩 + LLM 응답)
```

### 4.3 Post-Retrieval의 장점

- **최대 비용 절감**: Vector DB 검색 + LLM 호출 모두 건너뜀
- **최대 지연시간 감소**: 캐시 히트 시 전체 파이프라인 건너뜀
- **단순한 구현**: 캐시 히트 시 바로 응답 반환

### 4.4 Post-Retrieval의 치명적 문제점

#### 문제 1: 사용자 지시 무시

```
┌────────────────────────────────────────────────────────────────┐
│              Post-Retrieval Cache의 False Positive 시나리오      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  [쿼리 A] "프랑스 혁명을 10단어로 설명해"                          │
│     → LLM 응답: "왕정 붕괴, 시민혁명, 자유평등 선언"                │
│     → 캐시에 저장됨                                              │
│                                                                │
│  [쿼리 B] "프랑스 혁명을 100단어로 상세히 설명해"                    │
│     → 임베딩 유사도: 0.92 (높음!)                                 │
│     → 캐시 히트! → "왕정 붕괴, 시민혁명, 자유평등 선언"             │
│     → ❌ 사용자는 100단어를 원했는데 10단어 응답을 받음              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

#### 문제 2: 대화 컨텍스트 오류

MeanCache 논문에서 검증된 문제:

```
[대화 1]
  Q1: "Python으로 선 그리기" → 응답 캐시됨
  Q2: "색상을 빨간색으로 바꿔" → 응답 캐시됨 (선의 색상)

[대화 2]
  Q3: "Python으로 원 그리기" → 캐시 미스, 새 응답
  Q4: "색상을 빨간색으로 바꿔" → ⚠️ Q2와 유사도 높음!
      → 캐시 히트 → ❌ "선"의 색상 코드를 반환
      → 사용자는 "원"의 색상을 바꾸려 했음
```

#### 문제 3: 정량적 False Hit 데이터

MeanCache 논문 (1,000 쿼리, 700 고유 + 300 중복 테스트):

| 시스템 | 일반 쿼리 False Hit | 대화 컨텍스트 False Hit | Precision |
|:------:|:-------------------:|:----------------------:|:---------:|
| GPTCache | 233/700 (33%) | 54건 | 0.52 |
| MeanCache | 89/700 (13%) | 3건 | 0.72 |

GPTCache의 경우 **고유 쿼리의 1/3이 잘못된 캐시 응답**을 받았다.

---

## 5. Pre-Retrieval vs Post-Retrieval 비교

### 5.1 비교 표

| 항목 | Pre-Retrieval Cache | Post-Retrieval Cache |
|------|:-------------------:|:--------------------:|
| **캐시 대상** | 문서 청크 (검색 결과) | LLM 최종 응답 |
| **LLM 호출** | 매번 호출 | 캐시 히트 시 건너뜀 |
| **사용자 지시 반영** | ✅ 항상 반영 | ❌ 무시될 수 있음 |
| **False positive 영향** | 낮음 (LLM이 필터링 가능) | 높음 (잘못된 응답 그대로 전달) |
| **비용 절감** | 중간 (Vector DB만 절약) | 높음 (Vector DB + LLM 절약) |
| **지연시간 절감** | 중간 | 최대 |
| **정확도 리스크** | 낮음 | 높음 |
| **대표 구현** | HuggingFace Cookbook | GPTCache |

### 5.2 의사결정 가이드

```
                    의사결정 플로우차트

    [사용자 쿼리 유형 분석]
            │
            ▼
    ┌───────────────────┐
    │ 동일 질문이 반복?    │
    │ (FAQ, 반복 문의)    │
    └───────┬───────────┘
            │
     Yes ───┤──── No
     │      │         │
     ▼      │         ▼
  Post-     │    ┌────────────────────┐
  Retrieval │    │ 사용자 지시가 다양?   │
  Cache     │    │ (형식, 톤, 길이 등)  │
  적합      │    └───────┬────────────┘
            │            │
            │     Yes ───┤──── No
            │     │      │         │
            │     ▼      │         ▼
            │  Pre-      │    어느 쪽이든
            │  Retrieval │    가능
            │  Cache     │
            │  적합      │
            │            │
            ▼            ▼
    ┌────────────────────────────┐
    │  조합 전략 (Hybrid)          │
    │  Pre-Retrieval: 기본 적용    │
    │  Post-Retrieval: 유사도     │
    │  > 0.95일 때만 적용          │
    └────────────────────────────┘
```

### 5.3 시나리오별 추천

| 시나리오 | 추천 캐시 | 이유 |
|---------|----------|------|
| 고객센터 FAQ 봇 ("배송 며칠 걸려요?") | Post-Retrieval | 같은 질문 반복, 형식 변화 없음 |
| CS 챗봇 ("환불해줘" → "영어로 설명해줘") | Pre-Retrieval | 같은 정보, 다른 출력 형식 |
| 다국어 지원 ("RAG란?" 한국어/영어) | Pre-Retrieval | 검색 대상 동일, 응답 언어 다름 |
| 내부 문서 검색 (높은 정확도 요구) | Pre-Retrieval | False positive 최소화 |
| 코드 생성 (동일 코드 반복 요청) | Post-Retrieval | 정확한 재현 필요 |

---

## 6. 캐시 인프라 옵션

### 6.1 시맨틱 캐시에 사용 가능한 인프라

시맨틱 캐시는 벡터 유사도 검색이 필요하므로, 일반 Redis(exact match)로는 불가능하다.

| 인프라 | 유형 | 특징 |
|--------|------|------|
| **Redis Stack** | In-memory + Vector | RediSearch 모듈로 벡터 검색 지원. 일반 Redis와 다름 |
| **FAISS** | 라이브러리 | Meta 개발, 로컬 벡터 인덱스. GPTCache/HuggingFace 기본 사용 |
| **Pinecone** | Managed Vector DB | 서버리스, 자동 스케일링 |
| **Qdrant** | Vector DB | Rust 기반, 고성능, 필터링 지원 |
| **Milvus** | Distributed Vector DB | Zilliz 개발, GPTCache와 같은 회사 |
| **Weaviate** | Vector DB | GraphQL API, 하이브리드 검색 |
| **ChromaDB** | Embedded Vector DB | 로컬 개발 최적, Python 네이티브 |
| **pgvector** | PostgreSQL 확장 | 기존 PostgreSQL에 벡터 검색 추가 |
| **MongoDB Atlas** | Document DB + Vector | Atlas Vector Search로 벡터 검색 지원 |
| **Upstash Vector** | Serverless Vector | Upstash Semantic Cache 공식 백엔드 |

### 6.2 주의: Redis vs Redis Stack

```
┌──────────────────────────────────────────────────────────────┐
│  일반 Redis                    Redis Stack                    │
│  ─────────                    ───────────                    │
│  Key-Value 저장               Key-Value + 벡터 검색           │
│  Exact match만 가능            시맨틱 유사도 검색 가능           │
│  GET/SET                      FT.SEARCH + KNN               │
│  캐시 키 = 쿼리 문자열          캐시 키 = 쿼리 임베딩 벡터        │
│                                                              │
│  "로그인 안됨" ≠ "로그인 안돼"   "로그인 안됨" ≈ "로그인 안돼"    │
│       ❌ 캐시 미스                    ✅ 캐시 히트             │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. GPTCache의 유사도 임계값 한계

### 7.1 고정 임계값의 딜레마

GPTCache 논문에서 확인된 한계:

> "Thresholds below 0.8 led to higher cache hit rates but also introduced irrelevant matches, decreasing the positive hit rate. Thresholds above 0.8 reduced cache hit rates significantly."

```
Cache Hit Rate vs Accuracy

Hit Rate  100% ┤
               │  ×
          80%  ┤     ×
               │        ×
          60%  ┤           ×
               │              ×
          40%  ┤                 ×
               │
          20%  ┤
               │
           0%  ┤────┬────┬────┬────┬────
              0.7  0.75  0.8  0.85  0.9   Threshold

Accuracy  100% ┤                       ×
               │                  ×
          95%  ┤             ×
               │        ×
          90%  ┤   ×
               │
          85%  ┤────┬────┬────┬────┬────
              0.7  0.75  0.8  0.85  0.9   Threshold

→ Hit rate와 Accuracy는 트레이드오프 관계
→ 0.8 부근이 균형점이나, 도메인별 튜닝 필요
```

### 7.2 추가 한계

- **고정 임계값(0.8)이 모든 도메인에 적용 불가**: 의료/법률은 0.95+ 필요
- **대규모 데이터셋에서 성능 저하**: 임베딩 셋이 커지면 동적 업데이트 시 지연 발생
- **TTL 기반 캐시 갱신은 빠르게 변하는 데이터에 부적합**: 주가, 실시간 뉴스 등
- **다중 턴 대화에서 컨텍스트 처리 미흡**: 이전 대화 맥락을 고려하지 못함
- **캐시 관리 자체의 리소스 오버헤드**: 제한된 환경에서 부담

---

## 8. 핵심 요약

### 8.1 잘못된 이해 (기존 문서)

- ❌ 3개 레이어(Embedding → Retrieval → Response)가 순차적으로 존재
- ❌ Embedding Cache가 별도 레이어로 의미 있음
- ❌ 각 레이어에 다른 TTL을 설정하면 최적화됨

### 8.2 올바른 이해 (근거 기반)

- ✅ 2가지 접근법(Pre-Retrieval vs Post-Retrieval)이 존재
- ✅ 임베딩 생성은 어떤 접근법이든 항상 필요 (건너뛸 수 없음)
- ✅ Pre-Retrieval은 안전하지만 LLM 비용 절감 없음
- ✅ Post-Retrieval은 비용 효율적이지만 False Positive 위험 (GPTCache 기준 33%)
- ✅ 시나리오에 따라 적합한 접근법이 다르며, Hybrid 전략도 가능

---

## Sources

- [HuggingFace Cookbook: Semantic Cache with FAISS](https://huggingface.co/learn/cookbook/en/semantic_cache_chroma_vector_database) — Pre-Retrieval 선택 근거, "loss of influence" 설명, 구현 코드
- [MeanCache: User-Centric Semantic Cache (arxiv)](https://arxiv.org/html/2403.02694v3) — GPTCache false hit 정량 데이터 (233/700), 대화 컨텍스트 문제
- [GPTCache GitHub](https://github.com/zilliztech/GPTCache) — Post-Retrieval 캐시 대표 구현
- [Creospan: Caching Patterns in RAG](https://creospan.com/caching-patterns-in-retrieval-augmented-generation/) — 5가지 RAG 캐싱 패턴 분류
- [Brain.co: Semantic Caching for RAG](https://brain.co/blog/semantic-caching-accelerating-beyond-basic-rag) — Pre vs Post 트레이드오프
- [Qdrant: Semantic Cache for AI](https://qdrant.tech/articles/semantic-cache-ai-data-retrieval/) — 캐시 배치 전략
- [GPT Semantic Cache 논문 (arxiv)](https://arxiv.org/html/2411.05276v3) — 임계값 한계, 성능 데이터
