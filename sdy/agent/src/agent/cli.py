"""CLI 인터페이스 — Typer 기반 명령행 도구."""

import typer

app = typer.Typer(help="Biomedical Research Agent")


@app.command()
def hello() -> None:
    """Smoke test."""
    typer.echo("Agent is ready!")


@app.command()
def ask(question: str = typer.Argument(help="질문")) -> None:
    """단일 질문을 에이전트에게 전달한다."""
    from agent.pipeline import ask_question

    typer.echo(f"Researching: {question}\n")
    result = ask_question(question)

    typer.echo(f"Answer:\n{result.answer}")
    if result.steps:
        typer.echo(f"\n(Used {result.total_steps} tool calls)")


@app.command()
def research(drug_name: str = typer.Argument(help="약물 이름")) -> None:
    """약물에 대한 종합 리포트를 생성한다."""
    from agent.pipeline import ask_question

    typer.echo(f"Researching {drug_name}...\n")

    query = (
        f"Research the drug '{drug_name}'. "
        f"1) Search PubMed for recent research articles. "
        f"2) Search ClinicalTrials.gov for active or recent trials. "
        f"3) Provide a comprehensive summary with citations."
    )
    result = ask_question(query, max_steps=15)

    typer.echo(f"Answer:\n{result.answer}")
    if result.steps:
        typer.echo(f"\n(Completed in {result.total_steps} steps)")


if __name__ == "__main__":
    app()
