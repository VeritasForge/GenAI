import pytest

from agent.tools.registry import ToolRegistry


def _dummy_tool(query: str) -> str:
    return f"result for {query}"


def _another_tool(pmid: str) -> str:
    return f"details for {pmid}"


class TestToolRegistry:
    """Tool Registry 테스트."""

    def test_should_register_and_list_tools(self):
        # Given
        registry = ToolRegistry()

        # When
        registry.register(
            name="search_pubmed",
            description="Search PubMed for articles",
            parameters={"query": "str", "max_results": "int (default: 5)"},
            func=_dummy_tool,
        )

        # Then
        assert "search_pubmed" in registry.tool_names

    def test_should_execute_registered_tool(self):
        # Given
        registry = ToolRegistry()
        registry.register(
            name="search_pubmed",
            description="Search PubMed",
            parameters={"query": "str"},
            func=_dummy_tool,
        )

        # When
        result = registry.execute("search_pubmed", {"query": "metformin"})

        # Then
        assert result == "result for metformin"

    def test_should_raise_when_executing_unregistered_tool(self):
        # Given
        registry = ToolRegistry()

        # When / Then
        with pytest.raises(KeyError, match="unknown_tool"):
            registry.execute("unknown_tool", {})

    def test_should_generate_prompt_description(self):
        # Given
        registry = ToolRegistry()
        registry.register(
            name="search_pubmed",
            description="Search PubMed for articles",
            parameters={"query": "str", "max_results": "int (default: 5)"},
            func=_dummy_tool,
        )
        registry.register(
            name="fetch_article",
            description="Get article details",
            parameters={"pmid": "str"},
            func=_another_tool,
        )

        # When
        description = registry.get_prompt_description()

        # Then
        assert "search_pubmed" in description
        assert "fetch_article" in description
        assert "Search PubMed for articles" in description
        assert "query" in description

    def test_should_raise_when_registering_duplicate_name(self):
        # Given
        registry = ToolRegistry()
        registry.register(
            name="tool_a",
            description="First tool",
            parameters={},
            func=_dummy_tool,
        )

        # When / Then
        with pytest.raises(ValueError, match="already registered"):
            registry.register(
                name="tool_a",
                description="Duplicate",
                parameters={},
                func=_another_tool,
            )
