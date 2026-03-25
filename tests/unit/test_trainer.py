"""Unit tests for the TrainerAgent."""

import json
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentforge.core.trainer import TrainerAgent, TrainingSession
from agentforge.research.cache import ResearchCache
from agentforge.research.mcp_discoverer import MCPDiscoverer, MCPServerSuggestion
from agentforge.research.result_parser import ResearchResultParser
from agentforge.research.web_searcher import WebSearcher
from agentforge.schemas.job import JobDefinition
from agentforge.schemas.research import ResearchResult, SearchResult
from agentforge.tools import create_default_registry
from agentforge.tools.registry import ToolRegistry


def make_job(**kwargs) -> JobDefinition:
    defaults = dict(
        raw_input="Analyze sales data",
        job_type="data_analysis",
        title="Analyze sales data",
        description="Analyze quarterly sales data and produce a summary report",
        domain="finance",
        output_type="text",
        expected_output="A report",
        tenant_id="test_tenant",
    )
    defaults.update(kwargs)
    return JobDefinition(**defaults)


def make_research(confidence: float = 0.8, **kwargs) -> ResearchResult:
    defaults = dict(
        job_type="data_analysis",
        domain="finance",
        required_skills=["data analysis", "financial reporting", "statistics"],
        required_tool_types=["web search", "calculator"],
        expert_approach=["Gather data", "Analyze trends", "Build report"],
        domain_knowledge_summary="Finance involves managing money and assets.",
        confidence=confidence,
    )
    defaults.update(kwargs)
    return ResearchResult(**defaults)


def make_trainer(
    registry: ToolRegistry = None,
    cached_research: ResearchResult = None,
    search_results: List[SearchResult] = None,
    parsed_research: ResearchResult = None,
    role_response: str = "Senior Data Analyst",
    criteria_response: list = None,
    mcp_suggestions: list = None,
) -> TrainerAgent:
    """Build a TrainerAgent with fully mocked dependencies."""
    if registry is None:
        registry = create_default_registry()

    mock_web_searcher = MagicMock(spec=WebSearcher)
    mock_web_searcher.research_job = AsyncMock(
        return_value=search_results or [
            SearchResult(url="https://example.com", title="Data Analysis Guide", snippet="...")
        ]
    )

    mock_cache = MagicMock(spec=ResearchCache)
    mock_cache.get = AsyncMock(return_value=cached_research)
    mock_cache.set = AsyncMock()

    _parsed = parsed_research or make_research()
    mock_parser = MagicMock(spec=ResearchResultParser)
    mock_parser.parse = AsyncMock(return_value=_parsed)
    mock_parser.refine = AsyncMock(return_value=make_research(confidence=0.6))

    mock_mcp = MagicMock(spec=MCPDiscoverer)
    mock_mcp.find_servers_for_capability = AsyncMock(
        return_value=mcp_suggestions or []
    )
    mock_mcp.save_suggestions = AsyncMock()

    # Mock LLMs
    def make_llm(content: str) -> MagicMock:
        m = MagicMock()
        resp = MagicMock()
        resp.content = content
        m.ainvoke = AsyncMock(return_value=resp)
        return m

    criteria = criteria_response or ["Criterion 1", "Criterion 2", "Criterion 3"]
    llm_fast = make_llm(json.dumps(criteria))
    # Second call for role extraction
    role_resp = MagicMock()
    role_resp.content = role_response
    criteria_resp = MagicMock()
    criteria_resp.content = json.dumps(criteria)
    llm_fast.ainvoke = AsyncMock(side_effect=[role_resp, criteria_resp])

    llm_planner = make_llm("Planning response")

    return TrainerAgent(
        tool_registry=registry,
        web_searcher=mock_web_searcher,
        result_parser=mock_parser,
        research_cache=mock_cache,
        mcp_discoverer=mock_mcp,
        llm_planner=llm_planner,
        llm_fast=llm_fast,
    )


@pytest.mark.asyncio
async def test_train_uses_cache_when_available():
    """If a high-confidence cache hit exists, web search should be skipped."""
    cached = make_research(confidence=0.85)
    trainer = make_trainer(cached_research=cached)
    session = await trainer.train(make_job())

    trainer.web_searcher.research_job.assert_not_called()
    assert session.research.confidence == 0.85
    assert any("cached" in line.lower() for line in session.training_log)


@pytest.mark.asyncio
async def test_train_runs_web_research_on_cache_miss():
    """When cache returns None, web research should be triggered."""
    trainer = make_trainer(cached_research=None)
    session = await trainer.train(make_job())

    trainer.web_searcher.research_job.assert_called_once()
    trainer.result_parser.parse.assert_called_once()


@pytest.mark.asyncio
async def test_train_logs_every_decision():
    """training_log must have entries for each of the 3 stages."""
    trainer = make_trainer(cached_research=None)
    session = await trainer.train(make_job())

    log_text = "\n".join(session.training_log)
    # Stage 1 — research
    assert "research" in log_text.lower() or "🔍" in log_text or "🌐" in log_text
    # Stage 2 — tool matching
    assert "tool" in log_text.lower() or "🔧" in log_text
    # Stage 3 — worker config
    assert "prompt" in log_text.lower() or "📝" in log_text


@pytest.mark.asyncio
async def test_train_includes_web_search_tool_always():
    """web_search must always be in the final tool set, even if not in requirements."""
    research = make_research(required_tool_types=["calculator"])  # no web_search required
    trainer = make_trainer(cached_research=None, parsed_research=research)
    session = await trainer.train(make_job())

    assert "web_search" in session.context_pack.tools


@pytest.mark.asyncio
async def test_train_handles_zero_confidence_research():
    """When research confidence < 0.5, Opus (PLANNER_MODEL) should be selected."""
    low_conf = make_research(confidence=0.3)
    # refine also returns low confidence so the model selection still picks PLANNER_MODEL
    trainer = make_trainer(
        cached_research=None,
        parsed_research=low_conf,
    )
    # Override refine to also return low confidence
    trainer.result_parser.refine = AsyncMock(return_value=make_research(confidence=0.35))
    session = await trainer.train(make_job())

    from agentforge.config.settings import settings
    assert session.context_pack.model == settings.PLANNER_MODEL


@pytest.mark.asyncio
async def test_system_prompt_contains_all_sections():
    """The generated system prompt must contain all required XML sections."""
    trainer = make_trainer(cached_research=None)
    session = await trainer.train(make_job())

    prompt = session.context_pack.system_prompt
    for tag in ["<role>", "<goal>", "<domain_knowledge>", "<required_skills>",
                "<expert_approach>", "<available_tools>", "<constraints>"]:
        assert tag in prompt, f"Missing tag: {tag}"


@pytest.mark.asyncio
async def test_mcp_discoverer_called_on_gaps():
    """When there are unmatched tool types, MCPDiscoverer should be called."""
    research = make_research(required_tool_types=["email sender", "sms gateway"])
    trainer = make_trainer(cached_research=None, parsed_research=research)
    session = await trainer.train(make_job())

    trainer.mcp_discoverer.find_servers_for_capability.assert_called()


@pytest.mark.asyncio
async def test_training_session_has_required_fields():
    """TrainingSession must have all expected fields populated."""
    trainer = make_trainer(cached_research=None)
    job = make_job()
    session = await trainer.train(job)

    assert session.job_id == job.job_id
    assert session.research is not None
    assert session.context_pack is not None
    assert len(session.training_log) > 0
    assert session.duration_seconds > 0
    assert len(session.context_pack.success_criteria) == 3
