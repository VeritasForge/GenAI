"""리포트 생성 — 수집된 데이터를 구조화된 약물 리포트로 변환한다."""

from agent.tools.types import Article, Trial


def format_report(
    drug_name: str,
    summary: str,
    articles: list[Article],
    trials: list[Trial],
) -> str:
    """수집된 데이터를 구조화된 리포트 텍스트로 변환한다.

    Args:
        drug_name: 약물 이름.
        summary: LLM이 생성한 요약문.
        articles: PubMed 논문 리스트.
        trials: 임상시험 리스트.

    Returns:
        포맷된 리포트 문자열.
    """
    sections = [
        f"# Drug Research Report: {drug_name}",
        "",
        "## Summary",
        summary,
        "",
        _format_articles_section(articles),
        _format_trials_section(trials),
    ]
    return "\n".join(sections)


def _format_articles_section(articles: list[Article]) -> str:
    """논문 섹션을 포맷한다."""
    lines = ["## Research Articles", ""]
    if not articles:
        lines.append("No articles found.")
        return "\n".join(lines)

    for i, article in enumerate(articles, 1):
        authors = ", ".join(article.authors[:3])
        if len(article.authors) > 3:
            authors += " et al."
        lines.append(f"### {i}. {article.title}")
        lines.append(f"- **PMID**: {article.pmid}")
        lines.append(f"- **Journal**: {article.journal} ({article.pub_date})")
        lines.append(f"- **Authors**: {authors}")
        if article.abstract:
            lines.append(f"- **Abstract**: {article.abstract[:200]}...")
        lines.append("")

    return "\n".join(lines)


def _format_trials_section(trials: list[Trial]) -> str:
    """임상시험 섹션을 포맷한다."""
    lines = ["## Clinical Trials", ""]
    if not trials:
        lines.append("No clinical trials found.")
        return "\n".join(lines)

    for i, trial in enumerate(trials, 1):
        conditions = ", ".join(trial.conditions) if trial.conditions else "N/A"
        interventions = ", ".join(trial.interventions) if trial.interventions else "N/A"
        lines.append(f"### {i}. {trial.title}")
        lines.append(f"- **NCT ID**: {trial.nct_id}")
        lines.append(f"- **Status**: {trial.status}")
        lines.append(f"- **Phase**: {trial.phase or 'N/A'}")
        lines.append(f"- **Conditions**: {conditions}")
        lines.append(f"- **Interventions**: {interventions}")
        lines.append("")

    return "\n".join(lines)
