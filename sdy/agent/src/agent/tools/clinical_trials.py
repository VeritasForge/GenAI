"""ClinicalTrials.gov API v2 래퍼.

임상시험 정보를 검색한다.
API 문서: https://clinicaltrials.gov/data-api/api
"""

import httpx

from agent.tools.types import Trial

_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def _http_get(url: str, params: dict) -> httpx.Response:
    """HTTP GET 요청. 테스트에서 mock하기 위해 분리."""
    return httpx.get(url, params=params, timeout=10.0)


def _parse_trial(study: dict) -> Trial:
    """API 응답의 개별 study를 Trial로 변환."""
    protocol = study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    conditions = protocol.get("conditionsModule", {})
    arms = protocol.get("armsInterventionsModule", {})

    phases = design.get("phases", [])
    interventions = arms.get("interventions", [])

    return Trial(
        nct_id=identification.get("nctId", ""),
        title=identification.get("briefTitle", ""),
        status=status.get("overallStatus", ""),
        phase=phases[0] if phases else "",
        conditions=conditions.get("conditions", []),
        interventions=[i["name"] for i in interventions],
    )


def search_trials(query: str, max_results: int = 5) -> list[Trial]:
    """ClinicalTrials.gov에서 임상시험을 검색한다.

    Args:
        query: 검색 쿼리.
        max_results: 최대 반환 결과 수.

    Returns:
        검색된 임상시험 리스트.

    Raises:
        ValueError: query가 비어있는 경우.
        RuntimeError: API 호출 실패 시.
    """
    if not query.strip():
        raise ValueError("query must not be empty")

    response = _http_get(
        _BASE_URL,
        params={
            "query.term": query,
            "pageSize": max_results,
            "format": "json",
        },
    )

    if response.status_code != 200:
        raise RuntimeError(f"ClinicalTrials API error: status {response.status_code}")

    studies = response.json().get("studies", [])
    return [_parse_trial(s) for s in studies]
