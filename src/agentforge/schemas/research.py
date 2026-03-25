"""Research schemas — the output of the Trainer's web research phase."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single result from a DuckDuckGo search, optionally with fetched page content."""

    url: str
    title: str
    snippet: str
    full_content: Optional[str] = None  # fetched and cleaned page text


class ResearchResult(BaseModel):
    """The structured output of the Trainer's web research phase.

    This is the central artifact that bridges raw web information and the Worker
    Agent's configuration. Every field in ContextPack ultimately traces back here.
    """

    job_type: str
    domain: str
    required_skills: List[str]
    # e.g. ["financial modeling", "data visualization", "regulatory knowledge"]
    required_tool_types: List[str]
    # e.g. ["spreadsheet", "chart generator", "document reader", "calculator"]
    expert_approach: List[str]
    # step-by-step: how an expert human would tackle this job
    domain_knowledge_summary: str
    # paragraph of key domain context the worker agent should know
    suggested_mcp_servers: List[str] = []
    # MCP package names that would help: e.g. "@modelcontextprotocol/server-sqlite"
    search_queries_used: List[str] = []
    sources: List[str] = []
    confidence: float = 0.0  # 0-1, how thorough the research was
    researched_at: datetime = Field(default_factory=datetime.utcnow)
