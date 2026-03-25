"""Unit tests for the web research engine."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentforge.research.cache import ResearchCache
from agentforge.research.result_parser import ResearchResultParser
from agentforge.research.web_searcher import WebSearcher
from agentforge.schemas.job import JobDefinition
from agentforge.schemas.research import ResearchResult, SearchResult


def make_job(**kwargs) -> JobDefinition:
    defaults = dict(
        raw_input="test input",
        job_type="data_analysis",
        title="Analyze data",
        description="Analyze sales data and produce a report",
        domain="finance",
        output_type="text",
        expected_output="A report",
        tenant_id="test_tenant",
    )
    defaults.update(kwargs)
    return JobDefinition(**defaults)


def make_research_result(**kwargs) -> ResearchResult:
    defaults = dict(
        job_type="data_analysis",
        domain="finance",
        required_skills=["data analysis", "reporting"],
        required_tool_types=["web search", "calculator"],
        expert_approach=["Gather data", "Analyze", "Report"],
        domain_knowledge_summary="Finance involves managing money.",
        confidence=0.8,
    )
    defaults.update(kwargs)
    return ResearchResult(**defaults)


# ── WebSearcher tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_returns_results():
    mock_results = [
        {"href": "https://example.com/1", "title": "Test 1", "body": "Body 1"},
        {"href": "https://example.com/2", "title": "Test 2", "body": "Body 2"},
    ]
    with patch("agentforge.research.web_searcher.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = mock_results
        searcher = WebSearcher(fetch_pages=False)
        results = await searcher.search("test query")

    assert len(results) == 2
    assert results[0].url == "https://example.com/1"
    assert results[0].title == "Test 1"
    assert results[1].snippet == "Body 2"


@pytest.mark.asyncio
async def test_page_fetch_cleans_html():
    html = """<html><body>
        <nav>Navigation stuff</nav>
        <footer>Footer stuff</footer>
        <p>Main content paragraph</p>
        <h2>Section heading</h2>
        <li>List item</li>
    </body></html>"""

    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )
        searcher = WebSearcher()
        content = await searcher.fetch_page("https://example.com")

    assert "Main content paragraph" in content
    assert "Navigation stuff" not in content
    assert "Footer stuff" not in content


@pytest.mark.asyncio
async def test_page_fetch_returns_empty_on_error():
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=Exception("network error")
        )
        searcher = WebSearcher()
        result = await searcher.fetch_page("https://broken.example.com")

    assert result == ""


# ── ResearchResultParser tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parser_extracts_skills():
    expected_data = {
        "required_skills": ["data analysis", "Python", "statistics"],
        "required_tool_types": ["web search", "calculator"],
        "expert_approach": ["Gather data", "Clean data", "Analyze", "Report"],
        "domain_knowledge_summary": "Finance domain knowledge.",
        "suggested_mcp_servers": [],
        "confidence": 0.8,
    }
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(expected_data)
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    parser = ResearchResultParser(llm=mock_llm)
    job = make_job()
    search_results = [
        SearchResult(url="https://example.com", title="Test", snippet="Finance skills needed")
    ]

    result = await parser.parse(search_results, job)

    assert "data analysis" in result.required_skills
    assert "web search" in result.required_tool_types
    assert result.confidence == 0.8
    assert len(result.expert_approach) == 4


@pytest.mark.asyncio
async def test_parser_handles_llm_failure_gracefully():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

    parser = ResearchResultParser(llm=mock_llm)
    job = make_job()
    result = await parser.parse([], job)

    assert result.confidence == 0.1
    assert "web search" in result.required_tool_types
    assert len(result.expert_approach) > 0


# ── ResearchCache tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_hit_returns_result(tmp_path):
    cache = ResearchCache(cache_dir=tmp_path, ttl_days=7)
    result = make_research_result()
    await cache.set("data_analysis", "finance", result)

    hit = await cache.get("data_analysis", "finance")
    assert hit is not None
    assert hit.job_type == "data_analysis"
    assert hit.confidence == 0.8


@pytest.mark.asyncio
async def test_cache_miss_returns_none(tmp_path):
    cache = ResearchCache(cache_dir=tmp_path, ttl_days=7)
    result = await cache.get("nonexistent", "domain")
    assert result is None


@pytest.mark.asyncio
async def test_cache_ttl_expired(tmp_path):
    cache = ResearchCache(cache_dir=tmp_path, ttl_days=1)
    result = make_research_result()

    # Write with an old timestamp
    import hashlib
    key = hashlib.sha256("data_analysis:finance".encode()).hexdigest()[:16]
    path = tmp_path / f"{key}.json"
    data = json.loads(result.model_dump_json())
    old_time = (datetime.utcnow() - timedelta(days=2)).isoformat()
    data["_cached_at"] = old_time
    path.write_text(json.dumps(data, default=str))

    hit = await cache.get("data_analysis", "finance")
    assert hit is None


@pytest.mark.asyncio
async def test_cache_invalidate(tmp_path):
    cache = ResearchCache(cache_dir=tmp_path, ttl_days=7)
    result = make_research_result()
    await cache.set("data_analysis", "finance", result)
    await cache.invalidate("data_analysis", "finance")

    hit = await cache.get("data_analysis", "finance")
    assert hit is None


@pytest.mark.asyncio
async def test_empty_search_results_handled():
    """Parser should handle empty search results gracefully."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "required_skills": [],
        "required_tool_types": [],
        "expert_approach": [],
        "domain_knowledge_summary": "",
        "suggested_mcp_servers": [],
        "confidence": 0.1,
    })
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    parser = ResearchResultParser(llm=mock_llm)
    job = make_job()
    result = await parser.parse([], job)

    assert "web search" in result.required_tool_types
    assert len(result.expert_approach) > 0
    assert result.confidence <= 0.1
