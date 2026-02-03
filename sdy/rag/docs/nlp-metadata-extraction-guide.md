# NLP 메타데이터 추출 가이드 — RAG 검색 품질 향상을 위한 LLM vs NER

> 이 문서는 RAG 파이프라인에서 NLP 기법을 활용한 메타데이터 추출 방법을 다룬다.
> 현재 프로젝트의 단순 벡터 검색에서 프로덕션 수준의 메타데이터 기반 검색으로 확장하기 위한 가이드이다.

---

## 목차

1. [NLP 추출이란](#1-nlp-추출이란)
2. [NLP 추출의 목적: 왜 필요한가](#2-nlp-추출의-목적-왜-필요한가)
3. [현재 RAG vs 메타데이터 Pre-filtering RAG](#3-현재-rag-vs-메타데이터-pre-filtering-rag)
4. [Pre-filtering vs Post-filtering](#4-pre-filtering-vs-post-filtering)
5. [NER 기반 메타데이터 추출](#5-ner-기반-메타데이터-추출)
6. [LLM 기반 메타데이터 추출](#6-llm-기반-메타데이터-추출)
7. [LLM vs NER 비교 및 선택 가이드](#7-llm-vs-ner-비교-및-선택-가이드)
8. [프로덕션 권장 아키텍처](#8-프로덕션-권장-아키텍처)
9. [Edge Cases & Caveats](#9-edge-cases--caveats)
10. [Sources](#10-sources)

---

## 1. NLP 추출이란

Ingestion Pipeline의 파싱 단계에서 문서를 단순히 텍스트로 읽어오는 것이 아니라, **자연어 처리(Natural Language Processing)** 기술을 활용해 구조화된 정보를 추출하는 것을 의미한다.

주요 NLP 추출 기법:

| 기법 | 설명 | 예시 |
|------|------|------|
| **개체명 인식(NER)** | 사람, 조직, 날짜, 금액 등 핵심 엔티티 자동 추출 | "삼성전자" → ORG |
| **키워드/핵심구 추출** | 문서의 주제를 대표하는 용어 식별 | "연차", "유연근무제" |
| **관계 추출** | 엔티티 간의 관계 파악 | "A사가 B사를 인수" |
| **문서 분류/요약** | 문서 유형 자동 분류 | "인사규정" → HR/policy |

---

## 2. NLP 추출의 목적: 왜 필요한가

핵심 목적은 **검색 품질(Retrieval Quality) 향상**이다. 기본 RAG는 벡터 유사도 검색 하나에만 의존하는데, 이것만으로는 한계가 있다. NLP 추출로 얻은 구조화된 정보가 그 한계를 보완한다.

### 2.1 메타데이터 필터링 (검색 전 범위 축소)

NER로 추출한 엔티티를 메타데이터로 저장하면, 벡터 검색 전에 필터를 걸 수 있다.

- 질문: "2024년 삼성 실적은?" → `date=2024`, `org=삼성` 필터 → 해당 문서들만 벡터 검색
- 전체 10만 청크를 검색하는 대신 수백 개로 좁혀서 정확도와 속도 모두 향상

### 2.2 하이브리드 검색 (벡터 + 키워드)

키워드/핵심구를 추출해두면 벡터 유사도 + 키워드 매칭을 결합할 수 있다. 벡터 검색이 놓치는 정확한 용어 매칭을 보완한다.

### 2.3 멀티홉 질문 대응 (관계 추출)

- 질문: "A사를 인수한 B사의 매출은?"
- 관계 추출로 `B사 →인수→ A사` 관계가 Knowledge Graph에 있으면, B사 매출 문서를 찾아갈 수 있음
- 단순 벡터 검색으로는 이런 간접적 연결을 찾기 어려움

### 2.4 검색 라우팅 (문서 분류)

문서 유형을 미리 분류해두면, 질문 성격에 따라 검색 대상을 지정할 수 있다:
- 법률 질문 → 법률 문서만 검색
- 기술 질문 → 기술 문서만 검색

### 요약

| NLP 기법 | 저장 형태 | 검색 시 활용 |
|----------|----------|-------------|
| NER | 메타데이터 (org, person, date) | 필터링 |
| 키워드 추출 | 인덱스 필드 | 키워드 매칭 |
| 관계 추출 | Knowledge Graph | 멀티홉 추론 |
| 문서 분류 | 메타데이터 (category) | 검색 라우팅 |

> **"문서를 넣을 때 미리 구조화해두면, 찾을 때 더 정확하게 찾는다"**

---

## 3. 현재 RAG vs 메타데이터 Pre-filtering RAG

### 3.1 현재 RAG의 검색 흐름

현재 `store.py:75`와 `retriever.py:29`의 흐름:

```
사용자 질문 "2024년 인사 규정 중 연차 정책은?"
    |
    v
query embedding (384차원 벡터)
    |
    v
ChromaDB.query(query_embeddings=[...], n_results=5)
    |
    v
코사인 유사도 상위 5개 Document 반환  <-- 10만 개 청크 전부 비교
```

**문제**: "2024년", "인사", "연차"라는 구체적 조건이 있는데도, **10만 개 청크 전체**에서 벡터 유사도만으로 검색한다. 재무보고서나 기술문서의 청크가 우연히 유사한 벡터를 가지면 노이즈로 섞여 들어올 수 있다.

### 3.2 NLP 추출 -> 메타데이터 저장 -> Pre-filtering 메커니즘

**Ingestion 파이프라인 (저장 시)**:

```
원본 문서 (인사규정_2024.pdf)
    |
    v
[파싱] PDF -> 텍스트
    |
    v
[NLP 추출] NER + 키워드 + 문서분류
    |  +-- NER:  조직="인사팀", 날짜="2024"
    |  +-- 키워드: "연차", "병가", "출산휴가"
    |  +-- 분류:  department="HR", doc_type="policy"
    |
    v
[청킹] 500자 단위로 분할
    |
    v
[임베딩] 각 청크 -> 384차원 벡터
    |
    v
[저장] ChromaDB에 벡터 + 메타데이터 함께 저장
       metadata = {
           "department": "HR",
           "year": 2024,
           "doc_type": "policy",
           "keywords": "연차,병가,출산휴가",
           "source": "인사규정_2024.pdf"
       }
```

**Retrieval 파이프라인 (검색 시)**:

```
사용자 질문 "2024년 인사 규정 중 연차 정책은?"
    |
    +-- [쿼리 NLP 분석] <-- 질문에서도 메타데이터 추출
    |    year=2024, department="HR", keyword="연차"
    |
    v
ChromaDB.query(
    query_embeddings=[...],
    n_results=5,
    where={"$and": [              <-- 메타데이터 Pre-filtering
        {"department": "HR"},
        {"year": 2024}
    ]}
)
    |
    v
HR + 2024 조건에 맞는 200개 청크만 벡터 검색  <-- 10만 -> 200개로 축소
    |
    v
상위 5개 Document 반환 (정확도 대폭 향상)
```

### 3.3 구체적 예시: 기업 지식 관리 시스템

10,000개의 사내 문서가 있다고 가정:

| 문서 | NLP 추출 메타데이터 |
|------|-------------------|
| 인사규정_2024.pdf | `department=HR, year=2024, doc_type=policy` |
| 인사규정_2023.pdf | `department=HR, year=2023, doc_type=policy` |
| 서버아키텍처.docx | `department=Engineering, year=2024, doc_type=technical` |
| 재무보고_Q3.pdf | `department=Finance, year=2024, doc_type=report` |
| 보안가이드.pdf | `department=Engineering, year=2024, doc_type=policy` |

**시나리오별 검색 비교**:

| 질문 | 현재 RAG (벡터만) | 메타데이터 Pre-filtering RAG |
|------|------------------|---------------------------|
| "2024년 연차 규정" | 10,000문서 전체 벡터 검색 -> "연차"와 유사한 재무용어 섞일 수 있음 | `where={year:2024, dept:HR}` -> 50개만 벡터 검색 -> 정확한 결과 |
| "서버 보안 정책" | 전체 검색 -> HR 보안교육 문서가 섞일 수 있음 | `where={dept:Engineering, doc_type:policy}` -> 관련 문서만 검색 |
| "올해 재무 실적" | 전체 검색 | `where={dept:Finance, year:2024}` -> 재무문서만 검색 |

### 3.4 현재 코드에서의 확장 포인트

현재 `store.py`의 `search()` 메서드에 `where` 파라미터를 추가하면 바로 사용 가능하다:

```python
# 현재 (store.py:75)
results = self._collection.query(
    query_embeddings=[query_embedding],
    n_results=actual_k,
    include=["documents", "metadatas", "distances"],
)

# 확장 시 (개념)
results = self._collection.query(
    query_embeddings=[query_embedding],
    n_results=actual_k,
    where={"$and": [{"department": "HR"}, {"year": 2024}]},  # 추가
    include=["documents", "metadatas", "distances"],
)
```

---

## 4. Pre-filtering vs Post-filtering

| 기준 | Pre-filtering | Post-filtering |
|------|--------------|----------------|
| **순서** | 메타데이터 먼저 -> 벡터 검색 | 벡터 검색 먼저 -> 메타데이터 필터 |
| **정확도** | 높음 (노이즈 사전 제거) | 관련 문서 누락 가능 |
| **결과 수 보장** | 필터 범위가 좁으면 k개 미달 가능 | k개 요청 -> 필터 후 미달 가능 |
| **속도** | 필터 후 작은 셋에서 벡터 검색 -> 빠름 | 전체 벡터 검색 -> 느림 |
| **적합 케이스** | 메타데이터가 명확한 조건 (연도, 부서) | 조건이 모호한 탐색적 질문 |

ChromaDB의 `where` 절은 **Pre-filtering** 방식으로 작동한다.

> **주의**: Pre-filtering 후 남은 벡터가 전체의 1% 미만이면 HNSW 인덱스의 검색 정밀도가 크게 떨어질 수 있다.

---

## 5. NER 기반 메타데이터 추출

### 5.1 NER (Named Entity Recognition)이란

**Named Entity Recognition (개체명 인식)** — 텍스트에서 미리 정의된 카테고리(사람, 조직, 장소, 날짜, 금액 등)에 해당하는 단어/구절을 자동으로 식별하고 분류하는 NLP 기법이다.

```
입력: "삼성전자는 2024년 1월 서울에서 갤럭시 S24를 출시했다."
                                     |
                                NER 처리
                                     |
출력: ORG("삼성전자")  DATE("2024년 1월")  LOC("서울")  PRODUCT("갤럭시 S24")
```

spaCy의 기본 제공 엔티티 타입:

| 라벨 | 의미 | 예시 |
|------|------|------|
| `PERSON` | 사람 이름 | "김철수", "Elon Musk" |
| `ORG` | 조직/회사 | "삼성전자", "Google" |
| `GPE` | 국가/도시 | "서울", "미국" |
| `DATE` | 날짜/기간 | "2024년 1월", "지난주" |
| `MONEY` | 금액 | "100만원", "$500" |
| `PRODUCT` | 제품 | "iPhone 15" |

### 5.2 spaCy로 NER 구현

```python
import spacy

nlp = spacy.load("ko_core_news_sm")  # 한국어 모델

def extract_metadata_ner(text: str) -> dict:
    """NER로 문서에서 메타데이터를 추출한다."""
    doc = nlp(text)

    metadata = {}
    for ent in doc.ents:
        key = ent.label_.lower()  # "ORG" -> "org"
        if key not in metadata:
            metadata[key] = []
        metadata[key].append(ent.text)

    # 리스트를 쉼표 구분 문자열로 (ChromaDB 호환)
    return {k: ",".join(set(v)) for k, v in metadata.items()}

# 사용 예
text = "삼성전자는 2024년 1월 서울에서 신제품을 출시했다."
metadata = extract_metadata_ner(text)
# -> {"org": "삼성전자", "date": "2024년 1월", "gpe": "서울"}
```

### 5.3 GLiNER — Zero-shot NER

spaCy는 미리 정해진 엔티티 타입만 인식하지만, GLiNER는 **실행 시점에 원하는 엔티티 타입을 지정**할 수 있다.

> **Zero-shot**이란: 모델이 **한 번도 학습하지 않은 작업을 수행**하는 것.
> - **Zero-shot**: 예제 0개, 바로 수행 ("영어 배운 적 없지만 일단 풀어봐")
> - **Few-shot**: 예제 몇 개 보여주고 수행 ("여기 예제 3개 있어, 참고해서 풀어봐")
> - **Fine-tuning**: 대량 데이터로 학습 후 수행 ("교재 1000페이지 공부하고 풀어봐")

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_base")

text = "삼성전자는 2024년 1월 서울에서 갤럭시 S24를 출시했다."
labels = ["회사", "날짜", "도시", "제품명"]  # <-- 자유롭게 지정

entities = model.predict_entities(text, labels)
# -> [("삼성전자", "회사"), ("2024년 1월", "날짜"), ("서울", "도시"), ...]
```

GLiNER 특징:
- 500M 파라미터 미만, CPU에서도 구동 가능
- NAACL 2024 벤치마크에서 ChatGPT zero-shot NER 성능 능가
- 학습 없이 새로운 엔티티 타입 즉시 추출 가능

### 5.4 spaCy vs GLiNER 비교

| 기준 | spaCy | GLiNER |
|------|-------|--------|
| 엔티티 타입 | 고정 (학습된 타입만) | 자유 지정 (zero-shot) |
| 새 타입 추가 | 재학습 필요 (수백~수천 레이블) | 라벨 문자열 변경만 |
| 속도 | 매우 빠름 (ms 단위) | 빠름 (수십 ms) |
| 정확도 | fine-tuned 시 F1 90%+ | zero-shot F1은 다소 낮음 |
| 한국어 지원 | `ko_core_news_sm` 모델 (정확도 제한적) | 다국어 지원 |

---

## 6. LLM 기반 메타데이터 추출

LLM에게 "이 문서에서 다음 정보를 추출해줘"라고 프롬프트를 보내고, **Structured Output (구조화된 출력)** 으로 JSON을 받는 방식이다.

### 6.1 핵심 메커니즘

Pydantic 모델로 원하는 출력 스키마를 정의 -> LLM의 function calling 또는 structured output API로 강제:

```python
from pydantic import BaseModel
from openai import OpenAI

# Step 1: 추출할 메타데이터 스키마 정의
class DocumentMetadata(BaseModel):
    department: str        # "HR", "Engineering", "Finance"
    year: int              # 2024
    doc_type: str          # "policy", "report", "technical"
    keywords: list[str]    # ["연차", "병가"]
    organizations: list[str]  # ["삼성전자"]

# Step 2: LLM에 구조화된 출력 요청
client = OpenAI()

def extract_metadata_llm(text: str) -> dict:
    """LLM structured output으로 메타데이터를 추출한다."""
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "문서에서 메타데이터를 추출하라."},
            {"role": "user", "content": text}
        ],
        response_format=DocumentMetadata,  # <-- Pydantic 모델로 출력 강제
    )
    return response.choices[0].message.parsed.model_dump()

# 사용 예
text = "2024년 인사팀 연차 규정: 입사 1년 미만 직원의 연차는..."
metadata = extract_metadata_llm(text)
# -> {"department": "HR", "year": 2024, "doc_type": "policy",
#     "keywords": ["연차", "입사"], "organizations": []}
```

### 6.2 Self-Query Retriever 패턴

검색 시 사용자 쿼리에서도 LLM이 메타데이터를 자동 추출하여 필터를 생성한다:

```
사용자: "2024년 인사팀 연차 규정 알려줘"
         |
         v  LLM이 쿼리 분석
         |
    +----+----+
    |         |
semantic   metadata filter
 query:    {"department": "HR",
"연차 규정"  "year": 2024}
    |         |
    v         v
 embedding   where 절
    |         |
    +----+----+
         |
         v
  ChromaDB filtered vector search
```

### 6.3 NER과의 핵심 차이: "보이는 것" vs "이해하는 것"

같은 문서를 처리할 때:

```
문서: "삼성전자 인사팀은 2024년부터 유연근무제를 시행한다."
```

**NER (spaCy)** — "보이는 것"만 추출:
```python
{"org": "삼성전자", "date": "2024년"}
# "인사팀"은 ORG로 인식할 수도, 못할 수도 있음
# "유연근무제"는 엔티티 타입에 없으므로 추출 불가
# "department=HR"이라는 분류는 불가능 (추론 능력 없음)
```

**LLM** — "의미를 이해하고" 추출:
```python
{"department": "HR",           # <-- "인사팀" -> HR로 추론
 "year": 2024,                 # <-- "2024년부터" -> 정수로 변환
 "policy_type": "work_policy", # <-- "유연근무제" -> 카테고리 분류
 "organizations": ["삼성전자"],
 "keywords": ["유연근무제", "인사팀"]}
```

**NER은 텍스트에서 "엔티티를 찾는" 것이고, LLM은 텍스트를 "이해하고 분류하는" 것이다.**

---

## 7. LLM vs NER 비교 및 선택 가이드

### 7.1 상세 비교표

| 기준 | NER (spaCy / GLiNER) | LLM (GPT-4o / Claude) |
|------|---------------------|----------------------|
| **속도** | 밀리초 단위 (spaCy), 수십ms (GLiNER) | 수초 (API 왕복) |
| **비용** | 무료 (로컬 실행) | 토큰당 과금 ($) |
| **정확도 (fine-tuned)** | F1 90%+ (학습 데이터 있을 때) | F1 70-80% (zero-shot) |
| **유연성** | spaCy: 고정 타입만 / GLiNER: zero-shot 가능 | 어떤 형태든 추출 가능 |
| **스키마 변경** | spaCy: 재학습 필요 / GLiNER: 라벨 변경만 | 프롬프트 수정만 |
| **할루시네이션** | 없음 (결정론적) | 있음 (없는 엔티티 생성 가능) |
| **복잡한 추론** | 불가 (패턴 매칭) | 가능 ("이 문서는 HR 부서 관련") |
| **대량 처리** | 초당 수만 건 | 초당 수 건 |
| **프라이버시** | 로컬 처리 가능 | 외부 API 호출 필요 (보안 이슈) |

### 7.2 선택 가이드

```
메타데이터 추출 방식 선택
    |
    +-- 문서가 얼마나 많은가?
    |   +-- 수만~수십만 건 --> NER (비용/속도 필수)
    |   +-- 수백~수천 건 --> LLM도 가능
    |
    +-- 추출할 정보가 정형적인가?
    |   +-- 이름/날짜/조직/금액 등 표준 엔티티 --> NER
    |   +-- "부서 분류", "문서 유형", "요약" 등 추론 필요 --> LLM
    |
    +-- 보안/프라이버시 요구사항?
    |   +-- 민감 데이터, 외부 전송 불가 --> NER (로컬) 또는 GLiNER
    |   +-- 제한 없음 --> LLM 가능
    |
    +-- 프로토타입 vs 프로덕션?
        +-- 프로토타입 --> LLM (빠르게 검증)
        +-- 프로덕션 --> NER로 교체 권장 (Explosion 권장 전략)
```

---

## 8. 프로덕션 권장 아키텍처

### 8.1 Explosion(spaCy 제작사) 권장 전략

> "LLM으로 프로토타입을 만들고, LLM이 자신의 더 저렴하고 신뢰할 수 있는 대체품을 학습시키게 하라."

1. **초기**: LLM으로 빠르게 메타데이터 추출 로직 검증
2. **데이터 축적**: LLM이 추출한 결과를 학습 데이터로 활용
3. **교체**: spaCy/GLiNER 모델을 fine-tune하여 LLM 대체
4. **프로덕션**: NER 모델로 대량 처리 (비용 1/100, 속도 100x)

### 8.2 하이브리드 아키텍처

실제 프로덕션에서는 두 방식을 결합하는 것이 가장 효과적이다:

```
문서 Ingestion 시 (저장):
+----------------------------------------------+
|                                              |
|  원본 문서                                    |
|      |                                       |
|      +--> NER (spaCy/GLiNER) --> 정형 메타데이터  |  <-- 빠르고 저렴
|      |    org, person, date, money            |
|      |                                       |
|      +--> LLM --> 비정형 메타데이터              |  <-- 느리지만 정확
|           department, doc_type, summary       |
|                                              |
|      --> 병합하여 ChromaDB에 저장               |
|                                              |
+----------------------------------------------+

질문 시 (검색):
+----------------------------------------------+
|                                              |
|  사용자 질문                                   |
|      |                                       |
|      +--> NER --> 엔티티 추출 (빠른 필터)       |
|      +--> LLM --> 의도 분석 + 필터 생성         |
|                  (Self-Query Retriever)       |
|                                              |
|      --> where 절 + 벡터 검색                  |
|                                              |
+----------------------------------------------+
```

---

## 9. Edge Cases & Caveats

- **한국어 NER 정확도**: spaCy의 한국어 모델(`ko_core_news_sm`)은 영어 모델 대비 정확도가 낮다. 한국어 특화 NER이 필요하면 Pororo, KoNLPy 등 대안을 검토해야 한다
- **LLM 할루시네이션**: LLM이 문서에 없는 메타데이터를 "만들어낼" 수 있다. Structured output으로 스키마를 강제해도 값 자체의 정확성은 보장되지 않는다
- **NER fine-tuning 비용**: 커스텀 엔티티를 인식시키려면 수백~수천 개의 레이블 데이터가 필요하다. GLiNER는 이 문제를 zero-shot으로 완화하지만 도메인 특화 정확도는 fine-tuned 모델보다 낮다
- **비용 폭증**: LLM으로 10만 건 문서를 처리하면 GPT-4o 기준 수백~수천 달러가 소요될 수 있다. gpt-4o-mini 사용 시에도 NER 대비 100배 이상 비싸다
- **과도한 Pre-filtering**: 필터가 너무 좁으면 관련 문서를 놓칠 수 있다. 부서 간 공유 문서가 특정 department로만 태깅된 경우 누락 발생
- **메타데이터 스키마 일관성**: "2024" vs "2024년" vs 2024(int) 같은 타입 불일치가 있으면 필터가 작동하지 않는다
- **HNSW 효율 저하**: Pre-filtering 후 남은 벡터가 전체의 1% 미만이면 HNSW 인덱스의 검색 정밀도가 크게 떨어진다

---

## 10. Sources

### 공식 문서
1. [ChromaDB — Metadata Filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering)
2. [spaCy EntityRecognizer API](https://spacy.io/api/entityrecognizer)
3. [LlamaIndex — Structured Outputs](https://developers.llamaindex.ai/python/examples/structured_outputs/structured_outputs/)
4. [NVIDIA RAG Blueprint — Advanced Metadata Filtering](https://docs.nvidia.com/rag/2.3.0/custom-metadata.html)
5. [Databricks — Unstructured Data Pipeline for RAG](https://docs.databricks.com/gcp/en/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag)

### 1차 자료 (프레임워크/도구 제작사)
6. [Explosion — Against LLM Maximalism](https://explosion.ai/blog/against-llm-maximalism)
7. [Haystack — Automated Structured Metadata Enrichment](https://haystack.deepset.ai/cookbook/metadata_enrichment)
8. [Haystack — Extracting Metadata Filters from Queries](https://haystack.deepset.ai/blog/extracting-metadata-filter)
9. [deepset — Leveraging Metadata in RAG](https://www.deepset.ai/blog/leveraging-metadata-in-rag-customization)

### 학술 논문
10. [GLiNER: Generalist Model for NER (NAACL 2024)](https://arxiv.org/abs/2311.08526)
11. [PMC — Comparative analysis of LLMs for labeling entities](https://pmc.ncbi.nlm.nih.gov/articles/PMC11804004/)

### 기술 블로그
12. [DEV Community — Pre and Post Filtering in Vector Search](https://dev.to/volland/pre-and-post-filtering-in-vector-search-with-metadata-and-rag-pipelines-2hji)
13. [Zilliz — Metadata Filtering, Hybrid Search or Agent in RAG](https://zilliz.com/blog/metadata-filtering-hybrid-search-or-agent-in-rag-applications)
14. [Azure — Boost RAG Performance with Metadata Filters](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/boost-rag-performance-enhance-vector-search-with-metadata-filters-in-azure-ai-se/4208985)
15. [MyScale — Filtered Vector Search](https://www.myscale.com/blog/filtered-vector-search-in-myscale/)
16. [Dataquest — Metadata Filtering and Hybrid Search](https://www.dataquest.io/blog/metadata-filtering-and-hybrid-search-for-vector-databases/)
17. [GraphRAG — Named Entity Recognition](https://graphrag.com/reference/preparation/ner/)
18. [GLiNER GitHub](https://github.com/urchade/GLiNER)
19. [Codesphere — Extracting Metadata from PDFs with NER using spaCy](https://codesphere.com/articles/extracting-data-pdfs-named-entity-recognition-spacy)
