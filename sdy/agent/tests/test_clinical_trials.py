from unittest.mock import patch

import httpx
import pytest

from agent.tools.clinical_trials import search_trials
from agent.tools.types import Trial


class TestSearchTrials:
    """ClinicalTrials.gov 검색 테스트."""

    def test_should_return_list_of_trials(self):
        # Given
        mock_response = httpx.Response(
            200,
            json={
                "studies": [
                    {
                        "protocolSection": {
                            "identificationModule": {
                                "nctId": "NCT00000001",
                                "briefTitle": "Metformin Phase 3 Trial",
                            },
                            "statusModule": {
                                "overallStatus": "COMPLETED",
                            },
                            "designModule": {
                                "phases": ["PHASE3"],
                            },
                            "conditionsModule": {
                                "conditions": ["Diabetes Mellitus"],
                            },
                            "armsInterventionsModule": {
                                "interventions": [
                                    {"name": "Metformin"},
                                    {"name": "Placebo"},
                                ],
                            },
                        }
                    },
                    {
                        "protocolSection": {
                            "identificationModule": {
                                "nctId": "NCT00000002",
                                "briefTitle": "Metformin Dosage Study",
                            },
                            "statusModule": {
                                "overallStatus": "RECRUITING",
                            },
                            "designModule": {
                                "phases": ["PHASE2"],
                            },
                            "conditionsModule": {
                                "conditions": ["Type 2 Diabetes"],
                            },
                            "armsInterventionsModule": {
                                "interventions": [
                                    {"name": "Metformin 500mg"},
                                ],
                            },
                        }
                    },
                ]
            },
        )

        with patch("agent.tools.clinical_trials._http_get") as mock_get:
            mock_get.return_value = mock_response

            # When
            trials = search_trials("metformin", max_results=2)

        # Then
        assert len(trials) == 2
        assert all(isinstance(t, Trial) for t in trials)
        assert trials[0].nct_id == "NCT00000001"
        assert trials[0].title == "Metformin Phase 3 Trial"
        assert trials[0].status == "COMPLETED"
        assert trials[0].phase == "PHASE3"
        assert trials[0].conditions == ["Diabetes Mellitus"]
        assert trials[0].interventions == ["Metformin", "Placebo"]

    def test_should_return_empty_list_when_no_results(self):
        # Given
        mock_response = httpx.Response(200, json={"studies": []})

        with patch("agent.tools.clinical_trials._http_get") as mock_get:
            mock_get.return_value = mock_response

            # When
            trials = search_trials("nonexistent_drug_xyz")

        # Then
        assert trials == []

    def test_should_raise_when_query_is_empty(self):
        # When / Then
        with pytest.raises(ValueError, match="query"):
            search_trials("")

    def test_should_raise_on_api_error(self):
        # Given
        mock_response = httpx.Response(500)

        with patch("agent.tools.clinical_trials._http_get") as mock_get:
            mock_get.return_value = mock_response

            # When / Then
            with pytest.raises(RuntimeError, match="ClinicalTrials"):
                search_trials("metformin")

    def test_should_handle_missing_optional_fields(self):
        # Given - 선택 필드가 없는 경우
        mock_response = httpx.Response(
            200,
            json={
                "studies": [
                    {
                        "protocolSection": {
                            "identificationModule": {
                                "nctId": "NCT99999999",
                                "briefTitle": "Minimal Study",
                            },
                            "statusModule": {
                                "overallStatus": "UNKNOWN",
                            },
                        }
                    }
                ]
            },
        )

        with patch("agent.tools.clinical_trials._http_get") as mock_get:
            mock_get.return_value = mock_response

            # When
            trials = search_trials("minimal")

        # Then
        assert len(trials) == 1
        assert trials[0].phase == ""
        assert trials[0].conditions == []
        assert trials[0].interventions == []
