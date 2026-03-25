"""Eval tests for tool selection accuracy."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.evals.evaluators import ToolSelectionEvaluator


def make_llm(response_dict: dict) -> MagicMock:
    m = MagicMock()
    r = MagicMock()
    r.content = json.dumps(response_dict)
    m.ainvoke = AsyncMock(return_value=r)
    return m


@pytest.mark.asyncio
async def test_correct_tool_selection_scores_high():
    llm = make_llm({"score": 0.95, "appropriate": True, "feedback": "Calculator is correct for math"})
    evaluator = ToolSelectionEvaluator(llm=llm)
    result = await evaluator.evaluate(
        job_description="Calculate mortgage payments",
        tools_selected=["calculator"],
        required_tool_types=["calculator", "math"],
    )
    assert result["score"] >= 0.65
    assert result["appropriate"] is True


@pytest.mark.asyncio
async def test_wrong_tool_selection_scores_low():
    llm = make_llm({"score": 0.2, "appropriate": False, "feedback": "Wrong tools for calculation"})
    evaluator = ToolSelectionEvaluator(llm=llm)
    result = await evaluator.evaluate(
        job_description="Calculate mortgage payments",
        tools_selected=["web_search"],
        required_tool_types=["calculator"],
    )
    assert result["score"] < 0.65
    assert result["appropriate"] is False


@pytest.mark.asyncio
async def test_tool_selection_datasets_loadable():
    """Verify the tool selection dataset JSON is valid and loadable."""
    import json
    from pathlib import Path

    dataset_path = Path(__file__).parent / "datasets" / "tool_selection.json"
    assert dataset_path.exists(), "tool_selection.json dataset missing"
    data = json.loads(dataset_path.read_text())
    assert len(data) > 0
    for item in data:
        assert "id" in item
        assert "description" in item
        assert "must_include_tools" in item
