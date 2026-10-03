import ast
import math
import operator
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from .memory import recall


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)


class CalculateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(min_length=1, max_length=200)


class EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


TOOL_MODELS = {
    "search_knowledge": SearchArgs,
    "recall_memory": SearchArgs,
    "calculate": CalculateArgs,
    "current_time": EmptyArgs,
}
DESCRIPTIONS = {
    "search_knowledge": "搜索用户上传的知识库，返回带引用编号的原文片段。可换关键词再次检索。",
    "recall_memory": "搜索用户主动保存的跨会话偏好和事实记忆。",
    "calculate": "精确计算数学表达式，支持加减乘除、括号、余数和小整数幂。",
    "current_time": "获取当前 UTC 时间。",
}
SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": DESCRIPTIONS[name],
            "parameters": model.model_json_schema(),
        },
    }
    for name, model in TOOL_MODELS.items()
]


def calculate(expression):
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 60:
        raise ValueError("表达式过于复杂")
    binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 10 or right != int(right)):
                raise ValueError("幂指数必须为 -10 到 10 的整数")
            value = binary[type(node.op)](left, right)
        else:
            raise ValueError("只允许数字和数学运算符")
        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("数值超出范围")
        return value

    return visit(tree.body)


async def execute_tool(name, arguments, knowledge, store):
    if name not in TOOL_MODELS:
        raise ValueError(f"未知工具：{name}")
    args = TOOL_MODELS[name].model_validate(arguments)
    if name == "search_knowledge":
        return await knowledge.search(args.query)
    if name == "recall_memory":
        return recall(store, args.query)
    if name == "calculate":
        return {"expression": args.expression, "result": calculate(args.expression)}
    return {"utc": datetime.now(timezone.utc).isoformat()}
