"""Unit tests for the EpisodicMemory system."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentforge.core.evaluator import EvaluationResult
from agentforge.core.worker import WorkerResult
from agentforge.memory.episodic import EpisodicMemory, ExperienceRecord
from agentforge.schemas.job import JobDefinition


def make_job(**kwargs) -> JobDefinition:
    defaults = dict(
        raw_input="Analyze data",
        job_type="data_analysis",
        title="Analyze data",
        description="Analyze quarterly sales data",
        domain="finance",
        output_type="text",
        expected_output="A report",
        tenant_id="tenant_a",
    )
    defaults.update(kwargs)
    return JobDefinition(**defaults)


def make_experience(**kwargs) -> ExperienceRecord:
    defaults = dict(
        job_type="data_analysis",
        domain="finance",
        tenant_id="tenant_a",
        job_description_summary="Analyze quarterly sales data",
        research_queries_used=["https://example.com"],
        skills_discovered=["data analysis", "reporting"],
        tool_types_needed=["web search", "calculator"],
        tools_that_worked=["web_search", "calculator"],
        quality_score=0.85,
        research_confidence=0.8,
        lessons=["Use calculator for financial math", "Web search for market context"],
        approach_summary="Used web_search and calculator to analyze data",
        duration_seconds=12.5,
    )
    defaults.update(kwargs)
    return ExperienceRecord(**defaults)


def make_memory_with_mocks():
    """Create an EpisodicMemory with mocked Qdrant and embedder."""
    memory = EpisodicMemory.__new__(EpisodicMemory)
    memory.qdrant_url = "http://localhost:6333"
    memory.openai_api_key = ""

    # Mock Qdrant client
    mock_client = MagicMock()
    mock_client.upsert = AsyncMock()
    memory._client = mock_client

    # Mock embedder
    mock_embedder = MagicMock()
    mock_embedder.aembed_query = AsyncMock(return_value=[0.1] * 1536)
    memory._embedder = mock_embedder

    return memory


@pytest.mark.asyncio
async def test_remember_stores_record():
    """remember() should call qdrant upsert with a vector and payload."""
    memory = make_memory_with_mocks()
    record = make_experience()
    await memory.remember(record)
    memory._client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_remember_and_recall_returns_relevant():
    """recall() should return ExperienceRecord objects matching the query."""
    memory = make_memory_with_mocks()
    experience = make_experience()

    # Mock search results
    mock_hit = MagicMock()
    mock_hit.payload = json.loads(experience.model_dump_json())
    memory._client.search = AsyncMock(return_value=[mock_hit])

    job = make_job()
    results = await memory.recall(job)

    assert len(results) == 1
    assert results[0].job_type == "data_analysis"
    assert results[0].quality_score == 0.85


@pytest.mark.asyncio
async def test_high_quality_experience_skips_web_search():
    """TrainerAgent should skip web search when recall returns quality >= 0.8."""
    from agentforge.core.trainer import TrainerAgent
    from agentforge.research.cache import ResearchCache
    from agentforge.research.mcp_discoverer import MCPDiscoverer
    from agentforge.research.result_parser import ResearchResultParser
    from agentforge.research.web_searcher import WebSearcher
    from agentforge.tools import create_default_registry

    high_quality_exp = make_experience(quality_score=0.9)
    memory = make_memory_with_mocks()
    memory._client.search = AsyncMock(
        return_value=[MagicMock(payload=json.loads(high_quality_exp.model_dump_json()))]
    )

    registry = create_default_registry()
    mock_web_searcher = MagicMock(spec=WebSearcher)
    mock_web_searcher.research_job = AsyncMock(return_value=[])
    mock_cache = MagicMock(spec=ResearchCache)
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    mock_parser = MagicMock(spec=ResearchResultParser)
    mock_parser.parse = AsyncMock()
    mock_mcp = MagicMock(spec=MCPDiscoverer)
    mock_mcp.find_servers_for_capability = AsyncMock(return_value=[])
    mock_mcp.save_suggestions = AsyncMock()

    def make_llm(content=""):
        m = MagicMock()
        r = MagicMock()
        r.content = content or '["Role Title"]'
        m.ainvoke = AsyncMock(side_effect=[
            MagicMock(content="Role Title"),
            MagicMock(content='["Criterion 1", "Criterion 2", "Criterion 3"]'),
        ])
        return m

    trainer = TrainerAgent(
        tool_registry=registry,
        web_searcher=mock_web_searcher,
        result_parser=mock_parser,
        research_cache=mock_cache,
        mcp_discoverer=mock_mcp,
        llm_planner=make_llm(),
        llm_fast=make_llm(),
        episodic_memory=memory,
    )

    session = await trainer.train(make_job())

    # Web search should NOT have been called
    mock_web_searcher.research_job.assert_not_called()
    mock_parser.parse.assert_not_called()
    assert any("⚡" in line or "skipping" in line.lower() for line in session.training_log)


@pytest.mark.asyncio
async def test_lessons_extracted_from_evaluation():
    """extract_lessons() should call LLM and return a list of lesson strings."""
    memory = make_memory_with_mocks()

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(content='["Lesson 1: Use calculator", "Lesson 2: Search first"]')
    )

    job = make_job()
    worker_result = WorkerResult(output="Analysis complete", tools_called=["calculator"], success=True)
    evaluation = EvaluationResult(
        quality_score=0.85,
        criteria_met=["Complete"],
        criteria_failed=[],
        feedback="Good job",
        passed=True,
    )

    lessons = await memory.extract_lessons(job, worker_result, evaluation, mock_llm)

    assert len(lessons) == 2
    assert "Lesson 1: Use calculator" in lessons


@pytest.mark.asyncio
async def test_tenant_isolation_in_recall():
    """recall() should pass tenant_id filter to Qdrant."""
    memory = make_memory_with_mocks()
    memory._client.search = AsyncMock(return_value=[])

    job_a = make_job(tenant_id="tenant_a")
    job_b = make_job(tenant_id="tenant_b")

    await memory.recall(job_a)
    await memory.recall(job_b)

    calls = memory._client.search.call_args_list
    assert len(calls) == 2
    # Verify filter was passed (the search was called with query_filter)
    for call in calls:
        assert "query_filter" in call.kwargs or len(call.args) >= 4


@pytest.mark.asyncio
async def test_recall_returns_empty_when_no_client():
    """recall() should return [] gracefully when Qdrant is not available."""
    memory = EpisodicMemory.__new__(EpisodicMemory)
    memory._client = None
    memory._embedder = None

    results = await memory.recall(make_job())
    assert results == []
