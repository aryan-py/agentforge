"""Integration tests for the full AgentForge pipeline."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agentforge.core.evaluator import EvaluationResult, JobEvaluator
from agentforge.core.meta_agent import MetaAgent
from agentforge.core.pipeline import AgentForgePipeline
from agentforge.core.trainer import TrainerAgent, TrainingSession
from agentforge.core.worker import WorkerAgent, WorkerResult
from agentforge.research.cache import ResearchCache
from agentforge.research.mcp_discoverer import MCPDiscoverer
from agentforge.research.result_parser import ResearchResultParser
from agentforge.research.web_searcher import WebSearcher
from agentforge.schemas.context_pack import ContextPack
from agentforge.schemas.job import JobDefinition
from agentforge.schemas.research import ResearchResult, SearchResult
from agentforge.tools import create_default_registry


def make_job(**kwargs) -> JobDefinition:
    defaults = dict(
        raw_input="Analyze sales data",
        job_type="data_analysis",
        title="Analyze sales data",
        description="Analyze quarterly sales data and produce a summary report",
        domain="finance",
        output_type="text",
        expected_output="A report",
        tenant_id="test",
    )
    defaults.update(kwargs)
    return JobDefinition(**defaults)


def make_research() -> ResearchResult:
    return ResearchResult(
        job_type="data_analysis",
        domain="finance",
        required_skills=["data analysis", "reporting"],
        required_tool_types=["web search", "calculator"],
        expert_approach=["Gather data", "Analyze", "Report"],
        domain_knowledge_summary="Finance domain.",
        confidence=0.8,
    )


def make_context_pack() -> ContextPack:
    return ContextPack(
        system_prompt="<role>Analyst</role><goal>Analyze</goal><domain_knowledge>Finance</domain_knowledge><required_skills>1. Analysis</required_skills><expert_approach>1. Do it</expert_approach><available_tools>web_search</available_tools><constraints>Be accurate</constraints>",
        role="Data Analyst",
        goal="Analyze data",
        expert_approach=["Gather data", "Analyze", "Report"],
        tools=["web_search", "calculator"],
        knowledge_context="Finance domain.",
        success_criteria=["Complete", "Accurate", "Useful"],
    )


def make_pipeline() -> AgentForgePipeline:
    registry = create_default_registry()

    # Mock MetaAgent
    meta = MagicMock(spec=MetaAgent)
    meta.classify = AsyncMock(return_value=make_job())

    # Mock TrainerAgent
    trainer = MagicMock(spec=TrainerAgent)
    research = make_research()
    context_pack = make_context_pack()
    session = TrainingSession(
        job_id="test-job-id",
        research=research,
        context_pack=context_pack,
        match_result=registry.find_for_requirements(["web search"], "test"),
        training_log=["🌐 Starting research", "🔧 Matching tools", "✅ Training complete"],
        duration_seconds=1.5,
    )
    trainer.train = AsyncMock(return_value=session)

    # Mock WorkerAgent
    worker = MagicMock(spec=WorkerAgent)
    worker.execute = AsyncMock(
        return_value=WorkerResult(
            output="Here is the analysis of quarterly sales data...",
            tools_called=["web_search"],
            tool_call_count=1,
            success=True,
        )
    )

    # Mock JobEvaluator
    evaluator = MagicMock(spec=JobEvaluator)
    evaluator.evaluate = AsyncMock(
        return_value=EvaluationResult(
            quality_score=0.85,
            criteria_met=["Complete", "Accurate"],
            criteria_failed=[],
            feedback="Good output.",
            passed=True,
        )
    )

    return AgentForgePipeline(
        meta_agent=meta,
        trainer=trainer,
        worker=worker,
        evaluator=evaluator,
        tool_registry=registry,
    )


@pytest.mark.asyncio
async def test_full_pipeline():
    """All 4 nodes should run and the final state should be populated."""
    pipeline = make_pipeline()
    state = await pipeline.run_job("Analyze sales data", "test")

    assert state["status"] == "success"
    assert state["job"] is not None
    assert state["training_session"] is not None
    assert state["worker_result"] is not None
    assert state["evaluation"] is not None


@pytest.mark.asyncio
async def test_training_log_populated():
    """The training_session should contain a populated training_log."""
    pipeline = make_pipeline()
    state = await pipeline.run_job("Analyze sales data", "test")

    training_log = state["training_session"]["training_log"]
    assert len(training_log) >= 3
    assert any("research" in line.lower() or "🌐" in line for line in training_log)


@pytest.mark.asyncio
async def test_research_findings_in_context_pack():
    """Research skills should appear in the system prompt."""
    pipeline = make_pipeline()
    state = await pipeline.run_job("Analyze data", "test")

    # The context_pack was built with our mock research
    assert state["training_session"]["context_pack"] is not None
    assert state["training_session"]["research"]["required_skills"] is not None


@pytest.mark.asyncio
async def test_error_node_reached_on_failure():
    """When classify raises an exception, pipeline should route to handle_error."""
    pipeline = make_pipeline()
    pipeline.meta_agent.classify = AsyncMock(side_effect=Exception("Classification failed"))

    state = await pipeline.run_job("Bad input", "test")
    assert state["status"] == "failed"
