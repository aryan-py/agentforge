"""FastAPI application for AgentForge."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all singletons on startup."""
    fast_llm = ChatOpenAI(model=settings.ROUTER_MODEL, temperature=0)
    planner_llm = ChatOpenAI(model=settings.PLANNER_MODEL, temperature=0)

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

    app.state.pipeline = pipeline
    app.state.tool_registry = tool_registry
    app.state.research_cache = research_cache

    yield


app = FastAPI(title="AgentForge", version="0.1.0", lifespan=lifespan)

from agentforge.api.routes import health, jobs, memory, metrics  # noqa: E402

app.include_router(health.router)
app.include_router(jobs.router)
app.include_router(memory.router)
app.include_router(metrics.router)
