# Deep Research: RAG Reranking은 어디서, 어떻게 일어나는가?

## Executive Summary

Reranking은 **Vector DB에서 수행되는 것이 아니다**. Vector DB는 1단계(recall)만 담당하고, reranking은 **서비스 사이드의 별도 모델(Cross-Encoder)**이 수행한다. Vector DB의 벡터 유사도 검색과 Cross-Encoder의 reranking은 근본적으로 다른 메커니즘이며, 서로 독립적인 컴포넌트이다.

---

## Findings

### 1. Bi-Encoder(Vector DB)와 Cross-Encoder(Reranker)는 완전히 다른 모델이다

- **확신도**: `[Confirmed]`
- **출처**: [Pinecone — Rerankers and Two-Stage Retrieval](https://www.pinecone.io/learn/series/rag/rerankers/), [Weaviate — Cross-Encoders as Reranker](https://weaviate.io/blog/cross-encoders-as-reranker)
- **근거**: 모든 주요 출처에서 일관되게 설명

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Bi-Encoder (Vector DB)                            │
│                                                                     │
│   Query ──→ [Encoder] ──→ Query Vector ─┐                          │
│                                          ├──→ Cosine Similarity     │
│   Doc ────→ [Encoder] ──→ Doc Vector ───┘    (독립적으로 인코딩)     │
│                                                                     │
│   ✅ 빠름: 벡터 사전 계산, ANN 검색 <100ms                           │
│   ❌ 정확도: query-doc 간 세밀한 상호작용 포착 못함                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   Cross-Encoder (Reranker)                          │
│                                                                     │
│   [Query + Doc] ──→ [Transformer 전체 통과] ──→ 관련성 점수 (0~1)   │
│                      (query의 모든 토큰이 doc의 모든 토큰에 attend)   │
│                                                                     │
│   ✅ 정확도: 세밀한 의미 관계 포착                                    │
│   ❌ 느림: 문서 1개당 full forward pass 필요                          │
└─────────────────────────────────────────────────────────────────────┘
```

Bi-Encoder는 query와 document를 **각각 독립적으로** 벡터로 변환하여 비교한다. 반면 Cross-Encoder는 query와 document를 **함께 하나의 입력으로** Transformer에 넣어 모든 토큰 간의 attention을 계산한다. 그래서 정확하지만 느리다.

---

### 2. Reranking은 서비스 사이드에서 수행된다 (Vector DB 밖)

- **확신도**: `[Confirmed]`
- **출처**: [Oreate AI — Separation Deployment for Rerank](https://www.oreateai.com/blog/separation-deployment-scheme-for-rerank-and-embedding-models-in-ragflow/60c223938830e8217368c47b69b2f726), [ZeroEntropy — Neural Rerankers 101](https://www.zeroentropy.dev/articles/neural-rerankers-101)
- **근거**: 6개 이상의 독립 출처에서 동일한 아키텍처 설명

```
┌──────────────────────────────────────────────────────────────────────┐
│                    전체 파이프라인 흐름                                │
│                                                                      │
│  User Query                                                          │
│    │                                                                 │
│    ▼                                                                 │
│  ┌──────────────────────┐                                            │
│  │ 1. Bi-Encoder        │  쿼리를 벡터로 변환                         │
│  │    (Embedding 서비스)  │                                           │
│  └──────────┬───────────┘                                            │
│             │ query_vector                                           │
│             ▼                                                        │
│  ┌──────────────────────┐                                            │
│  │ 2. Vector DB (HNSW)  │  ANN 검색 + 메타데이터 필터링               │
│  │    top-20 recall      │  ← 여기까지가 Vector DB의 역할             │
│  └──────────┬───────────┘                                            │
│             │ 20개 문서 (텍스트 + 메타데이터 + 점수)                   │
│             ▼                                                        │
│  ┌──────────────────────────────────────────────────┐                │
│  │ 3. Reranker (Cross-Encoder)  ← 별도 GPU 서비스    │                │
│  │                                                    │               │
│  │  for each doc in top_20:                          │                │
│  │    score = cross_encoder(query, doc.text)          │               │
│  │                                                    │               │
│  │  결과: 20개 문서를 관련성 점수로 재정렬              │               │
│  │  top-5만 선별                                      │               │
│  └──────────────────────┬─────────────────────────────┘              │
│                          │ 5개 최고 관련성 문서                       │
│                          ▼                                           │
│  ┌──────────────────────┐                                            │
│  │ 4. LLM Generator     │  context + query → 답변 생성               │
│  └──────────────────────┘                                            │
└──────────────────────────────────────────────────────────────────────┘
```

**핵심**: Vector DB에서 나온 20개 문서의 **원본 텍스트**가 서비스 메모리로 로드되고, 이 텍스트가 Cross-Encoder 모델에 (query, doc_text) 쌍으로 전달된다. **Vector DB를 다시 호출하지 않는다.**

---

### 3. Reranker는 별도 GPU 마이크로서비스로 배포된다

- **확신도**: `[Confirmed]`
- **출처**: [Baseten — BEI High-throughput Inference](https://www.baseten.co/blog/how-we-built-bei-high-throughput-embedding-inference/), [Vast.ai — Serving Rerankers using vLLM](https://vast.ai/article/serving-rerankers-on-vastai-using-vllm)
- **근거**: 프로덕션 배포 사례 다수 확인

| 배포 방식 | 서빙 엔진 | 특징 |
| --------- | --------- | ---- |
| **vLLM** | GPU 서버 | 최근 reranker 서빙 지원 추가, OpenAI/Cohere API 호환 |
| **TensorRT-LLM / BEI** | GPU 서버 | 고처리량 임베딩/리랭킹 특화 |
| **Triton Inference Server** | GPU 서버 | NVIDIA 엔터프라이즈 표준 |
| **Xinference** | GPU 서버 | 오픈소스, 다양한 모델 지원 |
| **Cohere Rerank API** | 관리형 SaaS | API 호출만으로 사용, 100+ 언어 지원 |

프로덕션에서 권장하는 구조는 Embedding 서비스와 Reranker 서비스를 **별도 마이크로서비스로 분리** 배포하는 것이다.

---

### 4. 성능 Trade-off: 정확도 +20~35%, 레이턴시 +200~500ms

- **확신도**: `[Confirmed]`
- **출처**: [KnackForge — Reranking in RAG](https://knackforge.com/blog/rag), [Pinecone](https://www.pinecone.io/learn/series/rag/rerankers/)
- **근거**: 다수 벤치마크 결과 일치

| 지표 | Reranking 없음 | Reranking 적용 |
| ---- | -------------- | -------------- |
| Top-1 Precision | ~50% | 70~85% |
| 전체 정확도 향상 | baseline | +20~35% |
| 추가 레이턴시 | 0ms | +200~500ms (GPU), +400ms+ (CPU) |
| 권장 recall → rerank | - | top-20~50 → top-5~10 |

**실무 판단 기준**: LLM 생성이 2초+ 걸리면 reranking의 300~400ms는 무시 가능. end-to-end 목표가 1초 미만이면 GPU 가속이나 경량 모델(FlashRank, TinyBERT) 필요.

---

### 5. ColBERT: Bi-Encoder와 Cross-Encoder의 중간 지대

- **확신도**: `[Likely]`
- **출처**: [Medium — Cross-Encoders, ColBERT, and LLM-Based Rerankers](https://medium.com/@aimichael/cross-encoders-colbert-and-llm-based-re-rankers-a-practical-guide-a23570d88548)
- **근거**: 단일 가이드 기반이나 논리적으로 타당

```
속도                                              정확도
◄─────────────────────────────────────────────────────►

Bi-Encoder          ColBERT            Cross-Encoder
(가장 빠름)      (Late Interaction)     (가장 정확)
  │                   │                      │
  │  벡터 사전 계산    │  토큰별 벡터 계산      │  full attention
  │  단일 유사도 비교  │  MaxSim 연산          │  (query, doc) 쌍
  │  <100ms           │  ~수십ms              │  200~500ms
```

ColBERT는 document를 **토큰 단위 벡터 행렬**로 사전 계산하고, query도 토큰별 벡터로 변환한 뒤 **MaxSim** 연산으로 비교한다. Full cross-attention은 아니지만 토큰 수준 상호작용을 포착하여 bi-encoder보다 정확하고 cross-encoder보다 빠르다.

---

### 6. 일부 Vector DB는 내장 Reranker 모듈을 제공하지만, 본질은 동일

- **확신도**: `[Likely]`
- **출처**: [Weaviate — Cross-Encoders as Reranker](https://weaviate.io/blog/cross-encoders-as-reranker)
- **근거**: Weaviate 공식 블로그

Weaviate 같은 Vector DB는 reranker 모듈을 내장 제공한다. 하지만 이것은 **Vector DB 내부에서 벡터 검색을 다시 하는 것이 아니라**, 검색 결과에 대해 별도의 Cross-Encoder 모델을 호출하는 것이다. "내장"이라는 것은 API 호출을 한 번으로 줄여주는 편의성이지, 메커니즘이 다른 것은 아니다.

---

## Comparisons

| 기준 | Bi-Encoder (Vector DB) | Cross-Encoder (Reranker) | ColBERT (Late Interaction) |
| ---- | --------------------- | ----------------------- | ------------------------- |
| 실행 위치 | Vector DB | 별도 GPU 서비스 | 별도 GPU 서비스 |
| 입력 | query vector vs doc vector | (query_text, doc_text) 쌍 | query tokens vs doc token matrix |
| 사전 계산 | doc 벡터 사전 계산 ✅ | 불가 ❌ (쌍별 계산) | doc 토큰 벡터 사전 계산 ✅ |
| 속도 (20 docs) | <10ms | 200~500ms | ~50ms |
| 정확도 | 기본 | 최고 (+20~35%) | 중상 |
| 확장성 | 수억 벡터 가능 | top-K만 가능 (20~100) | 중간 |
| GPU 필요 | 검색 시 불필요 | 필수 (프로덕션) | 권장 |

**권장**: Bi-Encoder retrieval + Cross-Encoder reranking + LLM generation — 이 조합이 현재 엔터프라이즈 RAG의 표준 구성이다.

---

## Edge Cases & Caveats

- **Reranking이 해결 못하는 문제**: retrieval 자체가 나쁘면(잘못된 chunking, 인덱싱 누락 등) reranking을 추가해도 효과 없음. "deck chairs on the Titanic"이라는 표현이 쓰인다
- **CPU에서의 Reranking**: GPU 없이 CPU에서 Cross-Encoder를 돌리면 400ms+ 레이턴시. LLM 생성 시간이 길면 상대적으로 괜찮지만, 저지연 시스템에서는 병목
- **LLM-based Reranking**: GPT/Claude를 reranker로 쓰면 정확도는 최고지만 비용과 레이턴시가 비현실적. 고가치 쿼리(금융, 법률)에서만 적합

## Contradictions Found

- **"Reranking 정확도 향상폭"**: +20~35% (일부 출처) vs +70~90% (다른 출처) → 측정 지표가 다름(전체 정확도 vs top-1 precision). 두 수치 모두 유효하나 **측정 대상이 다르다**는 점에 주의

---

## Sources

1. [Rerankers and Two-Stage Retrieval — Pinecone](https://www.pinecone.io/learn/series/rag/rerankers/) — 공식 가이드
2. [Using Cross-Encoders as Reranker — Weaviate](https://weaviate.io/blog/cross-encoders-as-reranker) — 공식 블로그
3. [Cross-Encoders, ColBERT, and LLM-Based Rerankers — Michael Ryaboy](https://medium.com/@aimichael/cross-encoders-colbert-and-llm-based-re-rankers-a-practical-guide-a23570d88548) — 기술 블로그
4. [Neural Rerankers 101 — ZeroEntropy](https://www.zeroentropy.dev/articles/neural-rerankers-101) — 기술 블로그
5. [How We Built BEI — Baseten](https://www.baseten.co/blog/how-we-built-bei-high-throughput-embedding-inference/) — 프로덕션 사례
6. [Serving Rerankers on Vast.ai using vLLM](https://vast.ai/article/serving-rerankers-on-vastai-using-vllm) — 배포 가이드
7. [Separation Deployment for Rerank — Oreate AI](https://www.oreateai.com/blog/separation-deployment-scheme-for-rerank-and-embedding-models-in-ragflow/60c223938830e8217368c47b69b2f726) — 아키텍처 가이드
8. [Reranking in RAG: Accuracy Wins, Latency Costs — KnackForge](https://knackforge.com/blog/rag) — 벤치마크
9. [Inside the RAG Retrieval Pipeline — Mudassar Hakim](https://medium.com/@mudassar.hakim/inside-the-rag-retrieval-pipeline-bi-encoders-cross-encoders-re-rankers-two-stage-retrieval-c391bea7eae4) — 기술 블로그
10. [Beyond Simple Embeddings — WaterCrawl](https://watercrawl.dev/blog/Beyond-Simple-Embeddings) — 기술 블로그

---

## Research Metadata
- 검색 쿼리 수: 6 (일반 5 + SNS 1)
- 수집 출처 수: 10
- 출처 유형 분포: 공식 2, 블로그 6, 프로덕션 사례 2, SNS 0 (Reddit 결과 없음)
- 확신도 분포: Confirmed 4, Likely 2, Uncertain 0, Unverified 0
