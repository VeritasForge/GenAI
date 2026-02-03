from dataclasses import dataclass


@dataclass(frozen=True)
class Article:
    """PubMed 논문 정보를 담는 불변 데이터 클래스."""

    pmid: str
    title: str
    abstract: str
    authors: list[str]
    journal: str
    pub_date: str


@dataclass(frozen=True)
class Trial:
    """ClinicalTrials.gov 임상시험 정보를 담는 불변 데이터 클래스."""

    nct_id: str
    title: str
    status: str
    phase: str
    conditions: list[str]
    interventions: list[str]
