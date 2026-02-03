from unittest.mock import patch

import httpx
import pytest

from agent.tools.pubmed import fetch_article_details, search_pubmed
from agent.tools.types import Article


class TestSearchPubmed:
    """PubMed 검색 기능 테스트."""

    def test_should_return_list_of_articles_when_query_is_valid(self):
        # Given - PubMed API가 검색 결과를 반환하는 상황
        mock_search_response = httpx.Response(
            200,
            json={
                "esearchresult": {
                    "idlist": ["12345", "67890"],
                    "count": "2",
                }
            },
        )
        mock_fetch_response = httpx.Response(
            200,
            json={
                "result": {
                    "uids": ["12345", "67890"],
                    "12345": {
                        "uid": "12345",
                        "title": "Metformin and diabetes",
                        "authors": [
                            {"name": "Kim J"},
                            {"name": "Lee S"},
                        ],
                        "source": "Nature Medicine",
                        "pubdate": "2024 Jan",
                    },
                    "67890": {
                        "uid": "67890",
                        "title": "Aspirin clinical review",
                        "authors": [{"name": "Park H"}],
                        "source": "The Lancet",
                        "pubdate": "2024 Feb",
                    },
                }
            },
        )

        with patch("agent.tools.pubmed._http_get") as mock_get:
            mock_get.side_effect = [
                mock_search_response,
                mock_fetch_response,
            ]

            # When
            articles = search_pubmed("metformin", max_results=2)

        # Then
        assert len(articles) == 2
        assert all(isinstance(a, Article) for a in articles)
        assert articles[0].pmid == "12345"
        assert articles[0].title == "Metformin and diabetes"
        assert articles[0].authors == ["Kim J", "Lee S"]
        assert articles[0].journal == "Nature Medicine"
        assert articles[0].pub_date == "2024 Jan"

    def test_should_return_empty_list_when_no_results(self):
        # Given - 검색 결과가 없는 상황
        mock_search_response = httpx.Response(
            200,
            json={
                "esearchresult": {
                    "idlist": [],
                    "count": "0",
                }
            },
        )

        with patch("agent.tools.pubmed._http_get") as mock_get:
            mock_get.return_value = mock_search_response

            # When
            articles = search_pubmed("nonexistent_drug_xyz")

        # Then
        assert articles == []

    def test_should_raise_when_query_is_empty(self):
        # Given - 빈 쿼리
        # When / Then
        with pytest.raises(ValueError, match="query"):
            search_pubmed("")

    def test_should_raise_when_max_results_is_not_positive(self):
        # Given - 잘못된 max_results
        # When / Then
        with pytest.raises(ValueError, match="max_results"):
            search_pubmed("metformin", max_results=0)

    def test_should_raise_on_api_error(self):
        # Given - API가 에러를 반환하는 상황
        mock_response = httpx.Response(500)

        with patch("agent.tools.pubmed._http_get") as mock_get:
            mock_get.return_value = mock_response

            # When / Then
            with pytest.raises(RuntimeError, match="PubMed"):
                search_pubmed("metformin")


class TestFetchArticleDetails:
    """PubMed 논문 상세 조회 테스트."""

    def test_should_return_article_when_pmid_is_valid(self):
        # Given
        mock_response = httpx.Response(
            200,
            json={
                "result": {
                    "uids": ["12345"],
                    "12345": {
                        "uid": "12345",
                        "title": "Metformin and diabetes",
                        "authors": [
                            {"name": "Kim J"},
                            {"name": "Lee S"},
                        ],
                        "source": "Nature Medicine",
                        "pubdate": "2024 Jan",
                    },
                }
            },
        )

        with patch("agent.tools.pubmed._http_get") as mock_get:
            mock_get.return_value = mock_response

            # When
            article = fetch_article_details("12345")

        # Then
        assert isinstance(article, Article)
        assert article.pmid == "12345"
        assert article.title == "Metformin and diabetes"
        assert article.authors == ["Kim J", "Lee S"]
        assert article.journal == "Nature Medicine"

    def test_should_raise_when_pmid_is_empty(self):
        # When / Then
        with pytest.raises(ValueError, match="pmid"):
            fetch_article_details("")

    def test_should_raise_when_pmid_not_found(self):
        # Given - 존재하지 않는 PMID
        mock_response = httpx.Response(
            200,
            json={
                "result": {
                    "uids": ["99999"],
                    "99999": {"error": "cannot get document summary"},
                }
            },
        )

        with patch("agent.tools.pubmed._http_get") as mock_get:
            mock_get.return_value = mock_response

            # When / Then
            with pytest.raises(RuntimeError, match="not found"):
                fetch_article_details("99999")

    def test_should_handle_article_without_abstract(self):
        # Given - abstract가 없는 논문
        mock_response = httpx.Response(
            200,
            json={
                "result": {
                    "uids": ["11111"],
                    "11111": {
                        "uid": "11111",
                        "title": "Short communication",
                        "authors": [{"name": "Cho M"}],
                        "source": "BMJ",
                        "pubdate": "2024 Mar",
                    },
                }
            },
        )

        with patch("agent.tools.pubmed._http_get") as mock_get:
            mock_get.return_value = mock_response

            # When
            article = fetch_article_details("11111")

        # Then
        assert article.abstract == ""
