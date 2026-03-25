"""ContextPack — the complete Worker Agent configuration assembled from research."""

from typing import List, Literal, Optional

from pydantic import BaseModel


class ContextPack(BaseModel):
    """Everything the Worker Agent needs to execute a job.

    Assembled entirely from research findings by the TrainerAgent.
    No fields are hardcoded — every value is derived from web research
    or episodic memory.
    """

    system_prompt: str
    role: str
    goal: str
    expert_approach: List[str]  # from research — step by step plan
    tools: List[str]  # tool_ids to give the worker
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.1
    max_iterations: int = 15
    knowledge_context: str  # domain knowledge from research
    success_criteria: List[str]
    risk_level: Literal["low", "medium", "high"] = "low"
    research_confidence: float = 0.0
    estimated_cost_usd: Optional[float] = None
