"""파이프라인 오케스트레이션 — 에이전트와 Tool을 조합하여 워크플로우를 실행한다."""

from agent.core.agent import Agent
from agent.core.types import AgentResult
from agent.tools.clinical_trials import search_trials
from agent.tools.pubmed import fetch_article_details, search_pubmed
from agent.tools.registry import ToolRegistry


def create_default_registry() -> ToolRegistry:
    """기본 Tool 레지스트리를 생성한다."""
    registry = ToolRegistry()

    registry.register(
        name="search_pubmed",
        description="Search PubMed for medical research articles. "
        "Returns a list of articles with titles, abstracts, and metadata.",
        parameters={"query": "str", "max_results": "int (default: 5)"},
        func=search_pubmed,
    )

    registry.register(
        name="fetch_article",
        description="Get detailed information about a specific PubMed article by PMID.",
        parameters={"pmid": "str"},
        func=fetch_article_details,
    )

    registry.register(
        name="search_trials",
        description="Search ClinicalTrials.gov for clinical trials. "
        "Returns trials with status, phase, conditions, and interventions.",
        parameters={"query": "str", "max_results": "int (default: 5)"},
        func=search_trials,
    )

    return registry


def ask_question(query: str, max_steps: int = 10) -> AgentResult:
    """단일 질문을 에이전트에게 전달하여 답변을 받는다.

    Args:
        query: 사용자 질문.
        max_steps: 최대 에이전트 루프 횟수.

    Returns:
        에이전트 실행 결과.
    """
    registry = create_default_registry()
    agent = Agent(registry=registry)
    return agent.run(query, max_steps=max_steps)
