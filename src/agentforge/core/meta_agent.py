"""MetaAgent — classifies raw job requests into structured JobDefinitions."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentforge.schemas.job import JobDefinition

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a job classifier for an AI agent system.
Parse the given work request into a structured job definition.

Rules:
- job_type: snake_case verb phrase (e.g. "data_analysis", "report_generation", "code_review")
- domain: single lowercase word (e.g. "finance", "engineering", "hr", "marketing", "science")
- title: short human-readable title (max 8 words)
- output_type: one of "text", "structured", "file", "action"
- expected_output: what the final deliverable looks like (1 sentence)
- Be decisive — never leave fields empty.

Respond with valid JSON only:
{
  "job_type": "...",
  "title": "...",
  "description": "...",
  "domain": "...",
  "output_type": "text|structured|file|action",
  "expected_output": "...",
  "constraints": [],
  "priority": "low|medium|high|critical"
}"""


class MetaAgent:
    """Classifies raw user input into a structured JobDefinition.

    Uses a fast, cheap LLM (GPT-4o-mini) for classification since this is a
    straightforward structured extraction task.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    async def classify(self, raw_input: str, tenant_id: str) -> JobDefinition:
        """Parse a raw work request into a structured JobDefinition."""
        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=f"Work request: {raw_input}"),
                ]
            )
            import json

            raw = response.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
        except Exception as e:
            logger.warning(f"MetaAgent classification failed: {e}. Using fallback.")
            data = {
                "job_type": "general_task",
                "title": raw_input[:50],
                "description": raw_input,
                "domain": "general",
                "output_type": "text",
                "expected_output": "A completed response to the request",
                "constraints": [],
                "priority": "medium",
            }

        return JobDefinition(
            raw_input=raw_input,
            tenant_id=tenant_id,
            job_type=data.get("job_type", "general_task"),
            title=data.get("title", raw_input[:50]),
            description=data.get("description", raw_input),
            domain=data.get("domain", "general"),
            output_type=data.get("output_type", "text"),
            expected_output=data.get("expected_output", ""),
            constraints=data.get("constraints", []),
            priority=data.get("priority", "medium"),
        )
