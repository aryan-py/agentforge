"""LLM-as-judge evaluators for AgentForge quality assessment."""

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class OutputQualityEvaluator:
    """Evaluates worker output quality on completeness, accuracy, and usefulness."""

    def __init__(self, llm):
        self.llm = llm

    async def evaluate(self, job_description: str, output: str, skills: List[str]) -> Dict[str, Any]:
        """Score output 0-1 on completeness, accuracy, and usefulness."""
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = f"""Job: {job_description}
Output: {output[:800]}
Required skills: {', '.join(skills[:5])}

Score each dimension 0.0-1.0:
- completeness: did it address all parts?
- accuracy: does it appear factually correct?
- usefulness: is it practical and ready to use?

Return JSON only: {{"completeness": 0.0, "accuracy": 0.0, "usefulness": 0.0, "overall": 0.0}}"""

        try:
            resp = await self.llm.ainvoke([
                SystemMessage(content="You are an objective quality evaluator. Return only valid JSON."),
                HumanMessage(content=prompt),
            ])
            raw = resp.content.strip().strip("```json").strip("```").strip()
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"OutputQualityEvaluator failed: {e}")
            return {"completeness": 0.5, "accuracy": 0.5, "usefulness": 0.5, "overall": 0.5}


class ToolSelectionEvaluator:
    """Evaluates whether the trainer selected appropriate tools for the job."""

    def __init__(self, llm):
        self.llm = llm

    async def evaluate(
        self,
        job_description: str,
        tools_selected: List[str],
        required_tool_types: List[str],
    ) -> Dict[str, Any]:
        """Score tool selection accuracy."""
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = f"""Job: {job_description}
Tools selected: {tools_selected}
Tool types research said were needed: {required_tool_types}

Was the tool selection appropriate? Score 0.0-1.0.
Return JSON only: {{"score": 0.0, "appropriate": true, "feedback": "..."}}"""

        try:
            resp = await self.llm.ainvoke([
                SystemMessage(content="You evaluate AI tool selection. Return only valid JSON."),
                HumanMessage(content=prompt),
            ])
            raw = resp.content.strip().strip("```json").strip("```").strip()
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"ToolSelectionEvaluator failed: {e}")
            return {"score": 0.5, "appropriate": True, "feedback": "Evaluation failed"}


class ResearchQualityEvaluator:
    """Evaluates whether web research extracted relevant skills and knowledge."""

    def __init__(self, llm):
        self.llm = llm

    async def evaluate(
        self,
        job_description: str,
        skills_discovered: List[str],
        domain_knowledge: str,
        confidence: float,
    ) -> Dict[str, Any]:
        """Score the quality of research extraction."""
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = f"""Job: {job_description}
Skills discovered: {skills_discovered}
Domain knowledge: {domain_knowledge[:300]}
Confidence: {confidence}

Was this research thorough and relevant? Score 0.0-1.0.
Return JSON only: {{"score": 0.0, "relevant": true, "missing": [], "feedback": "..."}}"""

        try:
            resp = await self.llm.ainvoke([
                SystemMessage(content="You evaluate AI research quality. Return only valid JSON."),
                HumanMessage(content=prompt),
            ])
            raw = resp.content.strip().strip("```json").strip("```").strip()
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"ResearchQualityEvaluator failed: {e}")
            return {"score": 0.5, "relevant": True, "missing": [], "feedback": "Evaluation failed"}
