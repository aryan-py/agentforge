"""JobEvaluator — scores Worker output quality and extracts lessons."""

import json
import logging
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agentforge.core.worker import WorkerResult
from agentforge.schemas.job import JobDefinition
from agentforge.schemas.research import ResearchResult

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    """The output of a JobEvaluator.evaluate() call."""

    quality_score: float  # 0.0 - 1.0
    criteria_met: List[str]
    criteria_failed: List[str]
    feedback: str
    passed: bool  # quality_score >= 0.65


class JobEvaluator:
    """Evaluates the quality of Worker Agent output.

    Uses a fast LLM to score the output on completeness, accuracy, and quality.
    Also stores experience records for the EpisodicMemory system (Phase 2).
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    async def evaluate(
        self,
        job: JobDefinition,
        worker_result: WorkerResult,
        research: ResearchResult,
    ) -> EvaluationResult:
        """Score the worker's output against the job requirements."""

        if not worker_result.success or not worker_result.output:
            return EvaluationResult(
                quality_score=0.0,
                criteria_met=[],
                criteria_failed=["Worker failed to produce output"],
                feedback=worker_result.error or "No output produced",
                passed=False,
            )

        prompt = f"""Given this job: {job.description}

And this output:
{worker_result.output[:1000]}

The job required these skills: {', '.join(research.required_skills[:5])}

Score the output 0.0-1.0 on:
- Completeness: did it address all parts of the job?
- Accuracy: does it seem factually grounded?
- Quality: is it useful and well-structured?

Return JSON only:
{{
  "quality_score": 0.0,
  "criteria_met": ["..."],
  "criteria_failed": ["..."],
  "feedback": "..."
}}"""

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(
                        content="You are a quality evaluator for AI agent outputs. "
                        "Be objective and return only valid JSON."
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            raw = response.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())

            score = float(data.get("quality_score", 0.5))
            return EvaluationResult(
                quality_score=score,
                criteria_met=data.get("criteria_met", []),
                criteria_failed=data.get("criteria_failed", []),
                feedback=data.get("feedback", ""),
                passed=score >= 0.65,
            )
        except Exception as e:
            logger.warning(f"Evaluation LLM call failed: {e}")
            return EvaluationResult(
                quality_score=0.5,
                criteria_met=[],
                criteria_failed=[],
                feedback=f"Evaluation failed: {e}",
                passed=False,
            )
