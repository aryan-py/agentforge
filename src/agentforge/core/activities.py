"""Temporal activity definitions — each wraps a core AgentForge operation."""

import logging
from typing import Any

from temporalio import activity

logger = logging.getLogger(__name__)


@activity.defn
async def classify_job_activity(raw_input: str, tenant_id: str) -> dict:
    """Classify raw input into a structured JobDefinition dict."""
    from agentforge.config.settings import settings
    from agentforge.core.meta_agent import MetaAgent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=settings.ROUTER_MODEL, temperature=0)
    agent = MetaAgent(llm=llm)
    job = await agent.classify(raw_input, tenant_id)
    return job.model_dump(mode="json")


@activity.defn
async def train_agent_activity(job_dict: dict) -> dict:
    """Run the TrainerAgent — web research, tool matching, ContextPack assembly."""
    from agentforge.config.settings import settings
    from agentforge.core.trainer import TrainerAgent
    from agentforge.research.cache import ResearchCache
    from agentforge.research.mcp_discoverer import MCPDiscoverer
    from agentforge.research.result_parser import ResearchResultParser
    from agentforge.research.web_searcher import WebSearcher
    from agentforge.schemas.job import JobDefinition
    from agentforge.tools import create_default_registry
    from langchain_openai import ChatOpenAI

    fast_llm = ChatOpenAI(model=settings.ROUTER_MODEL, temperature=0)
    planner_llm = ChatOpenAI(model=settings.PLANNER_MODEL, temperature=0)

    trainer = TrainerAgent(
        tool_registry=create_default_registry(),
        web_searcher=WebSearcher(),
        result_parser=ResearchResultParser(llm=fast_llm),
        research_cache=ResearchCache(ttl_days=settings.RESEARCH_CACHE_TTL_DAYS),
        mcp_discoverer=MCPDiscoverer(),
        llm_planner=planner_llm,
        llm_fast=fast_llm,
    )
    job = JobDefinition(**job_dict)
    session = await trainer.train(job)
    return {
        "job_id": session.job_id,
        "research": session.research.model_dump(mode="json"),
        "context_pack": session.context_pack.model_dump(mode="json"),
        "training_log": session.training_log,
        "duration_seconds": session.duration_seconds,
    }


@activity.defn
async def request_approval_activity(job_dict: dict, training_dict: dict) -> None:
    """Notify that a high-risk job is awaiting human approval.

    In production this would send an email/Slack/webhook notification.
    Here we log the request and write to an approval queue file.
    """
    import json
    from pathlib import Path

    approval_file = Path("logs/pending_approvals.jsonl")
    approval_file.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "job_id": job_dict.get("job_id"),
        "title": job_dict.get("title"),
        "tenant_id": job_dict.get("tenant_id"),
        "risk_level": training_dict.get("context_pack", {}).get("risk_level"),
        "description": job_dict.get("description", "")[:200],
    }
    with approval_file.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.warning(
        f"HIGH-RISK JOB awaiting approval: {job_dict.get('job_id')} — {job_dict.get('title')}"
    )


@activity.defn
async def execute_job_activity(job_dict: dict, training_dict: dict) -> dict:
    """Execute the job using the WorkerAgent."""
    from agentforge.core.worker import WorkerAgent
    from agentforge.schemas.context_pack import ContextPack
    from agentforge.schemas.job import JobDefinition
    from agentforge.tools import create_default_registry

    job = JobDefinition(**job_dict)
    context_pack = ContextPack(**training_dict["context_pack"])
    registry = create_default_registry()
    worker = WorkerAgent()
    result = await worker.execute(job, context_pack, registry)
    return result.model_dump(mode="json")


@activity.defn
async def evaluate_result_activity(
    job_dict: dict, result_dict: dict, training_dict: dict
) -> dict:
    """Evaluate the worker's output quality."""
    from agentforge.config.settings import settings
    from agentforge.core.evaluator import JobEvaluator
    from agentforge.core.worker import WorkerResult
    from agentforge.schemas.job import JobDefinition
    from agentforge.schemas.research import ResearchResult
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=settings.ROUTER_MODEL, temperature=0)
    evaluator = JobEvaluator(llm=llm)

    job = JobDefinition(**job_dict)
    worker_result = WorkerResult(**result_dict)
    research = ResearchResult(**training_dict["research"])
    eval_result = await evaluator.evaluate(job, worker_result, research)
    return eval_result.model_dump()
