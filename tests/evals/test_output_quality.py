"""Eval tests for output quality using LLM-as-judge."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.evals.evaluators import OutputQualityEvaluator, ResearchQualityEvaluator


def make_llm(response_dict: dict) -> MagicMock:
    m = MagicMock()
    r = MagicMock()
    r.content = json.dumps(response_dict)
    m.ainvoke = AsyncMock(return_value=r)
    return m


@pytest.mark.asyncio
async def test_high_quality_output_scores_well():
    llm = make_llm({"completeness": 0.9, "accuracy": 0.85, "usefulness": 0.88, "overall": 0.88})
    evaluator = OutputQualityEvaluator(llm=llm)
    result = await evaluator.evaluate(
        job_description="Analyze sales trends for Q3",
        output="Based on my analysis, Q3 showed a 15% increase in sales...",
        skills=["data analysis", "reporting"],
    )
    assert result["overall"] >= 0.65
    assert "completeness" in result


@pytest.mark.asyncio
async def test_poor_output_scores_low():
    llm = make_llm({"completeness": 0.2, "accuracy": 0.3, "usefulness": 0.2, "overall": 0.23})
    evaluator = OutputQualityEvaluator(llm=llm)
    result = await evaluator.evaluate(
        job_description="Analyze sales trends for Q3",
        output="I don't know.",
        skills=["data analysis"],
    )
    assert result["overall"] < 0.65


@pytest.mark.asyncio
async def test_research_quality_evaluation():
    llm = make_llm({"score": 0.8, "relevant": True, "missing": [], "feedback": "Good research"})
    evaluator = ResearchQualityEvaluator(llm=llm)
    result = await evaluator.evaluate(
        job_description="Analyze EV market trends",
        skills_discovered=["market research", "trend analysis"],
        domain_knowledge="Electric vehicles are growing rapidly...",
        confidence=0.8,
    )
    assert result["score"] >= 0.65
    assert result["relevant"] is True


@pytest.mark.asyncio
async def test_evaluator_handles_llm_failure():
    """Evaluator should return fallback values when LLM fails."""
    m = MagicMock()
    m.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
    evaluator = OutputQualityEvaluator(llm=m)
    result = await evaluator.evaluate("test job", "test output", ["skill1"])
    assert "overall" in result
    assert 0.0 <= result["overall"] <= 1.0
