"""Unit tests for the ToolRegistry."""

import pytest
from langchain_core.tools import BaseTool

from agentforge.tools import create_default_registry
from agentforge.tools.registry import MatchResult, ToolRegistry, ToolRegistryEntry


def make_entry(tool_id: str, tool_types: list[str], **kwargs) -> ToolRegistryEntry:
    from langchain_core.tools import tool

    @tool
    def dummy_tool(input: str) -> str:
        """A dummy tool for testing."""
        return f"result: {input}"

    return ToolRegistryEntry(
        tool_id=tool_id,
        name=tool_id.replace("_", " ").title(),
        description="A test tool",
        tool_types=tool_types,
        langchain_tool=dummy_tool,
        **kwargs,
    )


def test_find_exact_match():
    registry = ToolRegistry()
    registry.register(make_entry("web_search", ["web search", "internet search"]))
    result = registry.find_for_requirements(["web search"], "*")
    assert len(result.matched) == 1
    assert result.matched[0].tool_id == "web_search"
    assert result.unmatched == []


def test_find_fuzzy_match():
    """'internet search' should fuzzy-match 'web search' via word overlap."""
    registry = ToolRegistry()
    registry.register(make_entry("web_search", ["web search", "internet search"]))
    result = registry.find_for_requirements(["internet search"], "*")
    assert len(result.matched) == 1
    assert result.matched[0].tool_id == "web_search"


def test_unmatched_returned():
    registry = ToolRegistry()
    registry.register(make_entry("web_search", ["web search"]))
    result = registry.find_for_requirements(["email sender"], "*")
    assert "email sender" in result.unmatched
    assert len(result.matched) == 0


def test_coverage_calculation():
    """3 matched out of 4 required = 0.75 coverage."""
    registry = ToolRegistry()
    registry.register(make_entry("web_search", ["web search"]))
    registry.register(make_entry("calculator", ["calculator", "math"]))
    registry.register(make_entry("file_reader", ["file reader"]))
    result = registry.find_for_requirements(
        ["web search", "calculator", "file reader", "email sender"], "*"
    )
    assert result.coverage == pytest.approx(0.75)
    assert len(result.matched) == 3
    assert "email sender" in result.unmatched


def test_resolve_returns_callable():
    registry = create_default_registry()
    tools = registry.resolve_tools(["web_search", "calculator"])
    assert len(tools) == 2
    for t in tools:
        assert isinstance(t, BaseTool)


def test_deduplicate_matched_tools():
    """One tool covering two requirements should appear only once in matched."""
    registry = ToolRegistry()
    registry.register(make_entry("web_search", ["web search", "research", "news"]))
    result = registry.find_for_requirements(["web search", "research"], "*")
    assert len(result.matched) == 1
    assert result.coverage == 1.0


def test_default_registry_initializes():
    registry = create_default_registry()
    assert len(registry.list_all_tool_types()) > 20


def test_tenant_restriction():
    registry = ToolRegistry()
    registry.register(make_entry("restricted", ["special tool"], allowed_tenants=["tenant_a"]))
    result_a = registry.find_for_requirements(["special tool"], "tenant_a")
    result_b = registry.find_for_requirements(["special tool"], "tenant_b")
    assert len(result_a.matched) == 1
    assert len(result_b.matched) == 0
