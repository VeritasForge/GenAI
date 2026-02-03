from agent.report import format_report
from agent.tools.types import Article, Trial


class TestFormatReport:
    """리포트 생성 테스트."""

    def test_should_format_report_with_articles_and_trials(self):
        # Given
        articles = [
            Article(
                pmid="12345",
                title="Metformin Study",
                abstract="A study about metformin.",
                authors=["Kim J", "Lee S"],
                journal="Nature",
                pub_date="2024 Jan",
            ),
        ]
        trials = [
            Trial(
                nct_id="NCT00000001",
                title="Metformin Phase 3",
                status="COMPLETED",
                phase="PHASE3",
                conditions=["Diabetes"],
                interventions=["Metformin", "Placebo"],
            ),
        ]

        # When
        report = format_report(
            drug_name="Metformin",
            summary="Metformin is widely used for diabetes.",
            articles=articles,
            trials=trials,
        )

        # Then
        assert "Metformin" in report
        assert "12345" in report
        assert "NCT00000001" in report
        assert "Metformin is widely used" in report

    def test_should_handle_empty_articles(self):
        # Given / When
        report = format_report(
            drug_name="Aspirin",
            summary="No articles found.",
            articles=[],
            trials=[],
        )

        # Then
        assert "Aspirin" in report
        assert "No articles found." in report

    def test_should_include_all_sections(self):
        # Given
        articles = [
            Article(
                pmid="111",
                title="Study A",
                abstract="Abstract A",
                authors=["Author"],
                journal="BMJ",
                pub_date="2024",
            ),
        ]
        trials = [
            Trial(
                nct_id="NCT111",
                title="Trial A",
                status="RECRUITING",
                phase="PHASE2",
                conditions=["Cancer"],
                interventions=["DrugX"],
            ),
        ]

        # When
        report = format_report(
            drug_name="DrugX",
            summary="Summary text.",
            articles=articles,
            trials=trials,
        )

        # Then - 주요 섹션이 포함되어야 함
        assert "Research Articles" in report or "논문" in report or "Articles" in report
        assert "Clinical Trials" in report or "임상시험" in report or "Trials" in report
