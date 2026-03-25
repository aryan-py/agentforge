#!/usr/bin/env python3
"""Phase 1 demo — runs 4 diverse jobs to showcase the research-first training system."""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from agentforge.config.settings import settings
from agentforge.core.evaluator import JobEvaluator
from agentforge.core.meta_agent import MetaAgent
from agentforge.core.pipeline import AgentForgePipeline
from agentforge.core.trainer import TrainerAgent
from agentforge.core.worker import WorkerAgent
from agentforge.research.cache import ResearchCache
from agentforge.research.mcp_discoverer import MCPDiscoverer
from agentforge.research.result_parser import ResearchResultParser
from agentforge.research.web_searcher import WebSearcher
from agentforge.tools import create_default_registry

DEMO_JOBS = [
    "Research the top 5 machine learning frameworks and compare their strengths, weaknesses, and best use cases",
    "Calculate the monthly payments on a $350,000 mortgage at 6.5% over 30 years, then show the full amortization for year 1",
    "Find 3 recent news articles about quantum computing breakthroughs and summarize the key scientific advances",
    "Write a Python function that sorts a list of employee records by salary descending, with unit tests",
]

WIDTH = 62


def box_line(text: str = "", fill: str = " ") -> str:
    return f"│  {text:<{WIDTH - 4}}│"


def separator(char: str = "─") -> str:
    return f"├{'─' * (WIDTH - 2)}┤"


def top() -> str:
    return f"┌{'─' * (WIDTH - 2)}┐"


def bottom() -> str:
    return f"└{'─' * (WIDTH - 2)}┘"


def print_result(state: dict, duration: float) -> None:
    job = state.get("job") or {}
    training = state.get("training_session") or {}
    research = training.get("research") or {}
    worker = state.get("worker_result") or {}
    evaluation = state.get("evaluation") or {}
    context_pack = training.get("context_pack") or {}

    print(top())
    print(box_line(f"JOB: {job.get('title', '')[:WIDTH - 8]}"))
    print(box_line(f"Domain: {job.get('domain', '')}"))
    print(separator())
    print(box_line("TRAINER RESEARCH LOG:"))
    for line in training.get("training_log", []):
        # Word-wrap long lines
        while len(line) > WIDTH - 6:
            print(box_line(f"  {line[:WIDTH - 8]}"))
            line = line[WIDTH - 8:]
        print(box_line(f"  {line}"))
    print(separator())
    skills = ", ".join(research.get("required_skills", [])[:4])
    tool_types = ", ".join(research.get("required_tool_types", [])[:4])
    tools_assigned = ", ".join(context_pack.get("tools", []))
    confidence = research.get("confidence", 0)
    print(box_line(f"SKILLS DISCOVERED: {skills[:WIDTH - 22]}"))
    print(box_line(f"TOOL TYPES NEEDED: {tool_types[:WIDTH - 22]}"))
    print(box_line(f"TOOLS ASSIGNED:    {tools_assigned[:WIDTH - 22]}"))
    print(box_line(f"RESEARCH CONFIDENCE: {confidence:.0%}"))
    print(separator())
    output = worker.get("output", state.get("error", "No output"))
    print(box_line("WORKER OUTPUT PREVIEW:"))
    preview = (output or "")[:300]
    while len(preview) > WIDTH - 6:
        print(box_line(f"  {preview[:WIDTH - 8]}"))
        preview = preview[WIDTH - 8:]
    if preview:
        print(box_line(f"  {preview}"))
    score = evaluation.get("quality_score", 0) or 0
    cost = 0.0
    print(box_line(f"Quality Score: {score:.0%}  |  Status: {state.get('status', '?')}"))
    print(box_line(f"Cost: ${cost:.4f}  |  Duration: {duration:.1f}s"))
    print(bottom())
    print()


async def main():
    print("\n" + "=" * WIDTH)
    print("  AgentForge — Phase 1 Demo: Research-First Agent Training")
    print("=" * WIDTH)
    print("  The Trainer searches the web to figure out what each job")
    print("  requires, then builds a Worker Agent from scratch.")
    print("=" * WIDTH + "\n")

    # Build singletons
    fast_llm = ChatOpenAI(model=settings.ROUTER_MODEL, temperature=0)
    planner_llm = ChatAnthropic(model=settings.PLANNER_MODEL, temperature=0)

    tool_registry = create_default_registry()
    web_searcher = WebSearcher()
    research_cache = ResearchCache(ttl_days=settings.RESEARCH_CACHE_TTL_DAYS)
    result_parser = ResearchResultParser(llm=fast_llm)
    mcp_discoverer = MCPDiscoverer()

    trainer = TrainerAgent(
        tool_registry=tool_registry,
        web_searcher=web_searcher,
        result_parser=result_parser,
        research_cache=research_cache,
        mcp_discoverer=mcp_discoverer,
        llm_planner=planner_llm,
        llm_fast=fast_llm,
    )
    worker = WorkerAgent()
    evaluator = JobEvaluator(llm=fast_llm)
    meta_agent = MetaAgent(llm=fast_llm)

    pipeline = AgentForgePipeline(
        meta_agent=meta_agent,
        trainer=trainer,
        worker=worker,
        evaluator=evaluator,
        tool_registry=tool_registry,
    )

    for i, job_desc in enumerate(DEMO_JOBS, 1):
        print(f"\n{'='*WIDTH}")
        print(f"  Job {i}/{len(DEMO_JOBS)}")
        print(f"{'='*WIDTH}")

        start = time.time()
        state = await pipeline.run_job(job_desc, tenant_id="demo")
        duration = time.time() - start

        print_result(state, duration)

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
