"""Job schemas for the AgentForge research-first pipeline."""

from datetime import datetime
from typing import Any, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class JobDefinition(BaseModel):
    """A structured job definition parsed from raw user input by the MetaAgent.

    This is the input to the training pipeline. The MetaAgent infers all fields
    from the raw natural-language request using a fast LLM.
    """

    job_id: str = Field(default_factory=lambda: str(uuid4()))
    raw_input: str
    job_type: str  # inferred by meta-agent, snake_case
    title: str
    description: str
    domain: str
    output_type: Literal["text", "structured", "file", "action"]
    expected_output: str
    constraints: List[str] = []
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    timeout_seconds: int = 300
    tenant_id: str
    requester_id: str = "api"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class JobResult(BaseModel):
    """The final output of a completed job run through the full pipeline.

    Captures not just the output but also what the trainer discovered during
    research, enabling post-hoc analysis of how the agent was configured.
    """

    job_id: str
    status: Literal["pending", "researching", "training", "running", "success", "failed", "review"]
    output: Optional[Any] = None
    quality_score: Optional[float] = None
    error: Optional[str] = None
    tools_used: List[str] = []
    research_queries: List[str] = []  # what the trainer searched for
    skills_discovered: List[str] = []  # what research found
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
