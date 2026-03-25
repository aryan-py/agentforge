"""WorkerAgent — executes jobs using a LangGraph ReAct agent built from a ContextPack."""

import asyncio
import logging
import traceback
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from agentforge.schemas.context_pack import ContextPack
from agentforge.schemas.job import JobDefinition
from agentforge.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class WorkerResult(BaseModel):
    """The output of a WorkerAgent.execute() call."""

    output: str = ""
    tools_called: List[str] = []
    tool_call_count: int = 0
    messages: List[dict] = []
    success: bool = True
    error: Optional[str] = None


class WorkerAgent:
    """Executes a job using a LangGraph ReAct agent configured by the TrainerAgent.

    The Worker receives a ContextPack (built entirely from web research) and
    executes the job by invoking a ReAct agent with the exact tools and system
    prompt the Trainer assembled.
    """

    async def execute(
        self,
        job: JobDefinition,
        context_pack: ContextPack,
        tool_registry: ToolRegistry,
    ) -> WorkerResult:
        """Execute a job using the trainer-built ContextPack."""

        # Step 1: Resolve tools
        tools = tool_registry.resolve_tools(context_pack.tools)
        if not tools:
            return WorkerResult(
                success=False,
                error="No tools could be resolved from context_pack.tools",
            )

        # Step 2: Build LangGraph ReAct agent
        llm = ChatOpenAI(
            model=context_pack.model,
            temperature=context_pack.temperature,
            max_tokens=4096,
        )
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=context_pack.system_prompt,
        )

        # Step 3: Execute with timeout
        try:
            state = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [HumanMessage(content=job.description)]},
                    config={"recursion_limit": context_pack.max_iterations},
                ),
                timeout=float(job.timeout_seconds),
            )
        except asyncio.TimeoutError:
            return WorkerResult(
                success=False,
                error=f"Job timed out after {job.timeout_seconds}s",
            )
        except Exception as e:
            logger.error(f"Worker execution error: {traceback.format_exc()}")
            return WorkerResult(success=False, error=str(e))

        # Step 4: Extract result
        messages = state.get("messages", [])
        output = ""
        tools_called: List[str] = []

        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content:
                output = msg.content if isinstance(msg.content, str) else str(msg.content)
            elif isinstance(msg, ToolMessage):
                # Extract tool name from the tool_call_id or name attribute
                tool_name = getattr(msg, "name", None) or "unknown_tool"
                tools_called.append(tool_name)

        serialized = []
        for m in messages:
            serialized.append({"type": type(m).__name__, "content": str(m.content)[:200]})

        return WorkerResult(
            output=output,
            tools_called=tools_called,
            tool_call_count=len(tools_called),
            messages=serialized,
            success=True,
        )
