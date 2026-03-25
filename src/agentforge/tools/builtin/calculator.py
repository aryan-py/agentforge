"""Built-in calculator tool — safe math evaluation."""

import ast
import math
import operator
from langchain_core.tools import tool

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_NAMES = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "pi": math.pi, "e": math.e, "ceil": math.ceil, "floor": math.floor,
    "pow": math.pow,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        op = _SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = _SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator")
        return op(_safe_eval(node.operand))
    elif isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name not in _SAFE_NAMES:
            raise ValueError(f"Function not allowed: {func_name}")
        args = [_safe_eval(a) for a in node.args]
        return _SAFE_NAMES[func_name](*args)
    elif isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]
        raise ValueError(f"Name not allowed: {node.id}")
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations safely.

    Use this tool for arithmetic, algebra, financial calculations, and statistics.
    Input should be a mathematical expression string, e.g. "350000 * 0.065 / 12".
    Returns the numeric result with the expression shown.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"


TOOL_TYPES = [
    "calculator",
    "math",
    "arithmetic",
    "computation",
    "financial calculation",
    "statistics",
    "number crunching",
    "formula evaluation",
]
