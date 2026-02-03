"""PubMed E-utilities API 래퍼.

PubMed의 ESearch + ESummary API를 사용하여 논문을 검색하고 상세 정보를 조회한다.
API 문서: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
"""

import httpx

from agent.tools.types import Article

_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _http_get(url: str, params: dict) -> httpx.Response:
    """HTTP GET 요청. 테스트에서 mock하기 위해 분리."""
    return httpx.get(url, params=params, timeout=10.0)


def _parse_article(data: dict) -> Article:
    """ESummary 응답의 개별 논문 데이터를 Article로 변환."""
    return Article(
        pmid=data["uid"],
        title=data.get("title", ""),
        abstract=data.get("abstract", ""),
        authors=[a["name"] for a in data.get("authors", [])],
        journal=data.get("source", ""),
        pub_date=data.get("pubdate", ""),
    )


def search_pubmed(query: str, max_results: int = 5) -> list[Article]:
    """PubMed에서 논문을 검색하여 Article 리스트를 반환한다.

    Args:
        query: 검색 쿼리 문자열.
        max_results: 최대 반환 결과 수 (기본값: 5).

    Returns:
        검색된 논문 리스트.

    Raises:
        ValueError: query가 비어있거나 max_results가 양수가 아닌 경우.
        RuntimeError: PubMed API 호출 실패 시.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if max_results < 1:
        raise ValueError("max_results must be positive")

    # 1단계: ESearch — 검색 쿼리로 PMID 리스트 조회
    search_response = _http_get(
        f"{_BASE_URL}/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
        },
    )
    if search_response.status_code != 200:
        raise RuntimeError(
            f"PubMed search API error: status {search_response.status_code}"
        )

    id_list = search_response.json()["esearchresult"]["idlist"]
    if not id_list:
        return []

    # 2단계: ESummary — PMID로 논문 상세 정보 조회
    return _fetch_summaries(id_list)


def fetch_article_details(pmid: str) -> Article:
    """PMID로 PubMed 논문 상세 정보를 조회한다.

    Args:
        pmid: PubMed 논문 ID.

    Returns:
        논문 정보가 담긴 Article 객체.

    Raises:
        ValueError: pmid가 비어있는 경우.
        RuntimeError: API 호출 실패 또는 논문을 찾을 수 없는 경우.
    """
    if not pmid.strip():
        raise ValueError("pmid must not be empty")

    articles = _fetch_summaries([pmid])
    if not articles:
        raise RuntimeError(f"Article not found: {pmid}")
    return articles[0]


def _fetch_summaries(pmids: list[str]) -> list[Article]:
    """PMID 리스트로 ESummary API를 호출하여 Article 리스트를 반환."""
    response = _http_get(
        f"{_BASE_URL}/esummary.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        },
    )
    if response.status_code != 200:
        raise RuntimeError(f"PubMed summary API error: status {response.status_code}")

    result = response.json()["result"]
    articles = []
    for pmid in result.get("uids", []):
        data = result[pmid]
        if "error" in data:
            raise RuntimeError(f"Article not found: {pmid}")
        articles.append(_parse_article(data))

    return articles
