"""LangGraph StateGraph orchestrating the full AgentForge pipeline."""

import logging
from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agentforge.core.evaluator import EvaluationResult, JobEvaluator
from agentforge.core.meta_agent import MetaAgent
from agentforge.core.trainer import TrainerAgent
from agentforge.core.worker import WorkerAgent
from agentforge.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentForgeState(TypedDict):
    """State passed between nodes in the AgentForge LangGraph pipeline."""

    messages: Annotated[list[BaseMessage], add_messages]
    job: Optional[dict]
    training_session: Optional[dict]  # serialized TrainingSession
    worker_result: Optional[dict]
    evaluation: Optional[dict]
    error: Optional[str]
    status: str
    tenant_id: str


class AgentForgePipeline:
    """The main LangGraph pipeline that orchestrates all 4 stages of AgentForge.

    Flow: classify → train → execute → evaluate
    Each node updates AgentForgeState and handles errors by routing to handle_error.
    """

    def __init__(
        self,
        meta_agent: MetaAgent,
        trainer: TrainerAgent,
        worker: WorkerAgent,
        evaluator: JobEvaluator,
        tool_registry: ToolRegistry,
        episodic_memory=None,
        llm_fast=None,
    ):
        self.meta_agent = meta_agent
        self.trainer = trainer
        self.worker = worker
        self.evaluator = evaluator
        self.tool_registry = tool_registry
        self.episodic_memory = episodic_memory
        self.llm_fast = llm_fast
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentForgeState)

        graph.add_node("classify", self._node_classify)
        graph.add_node("train", self._node_train)
        graph.add_node("execute", self._node_execute)
        graph.add_node("evaluate", self._node_evaluate)
        graph.add_node("handle_error", self._node_handle_error)

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify", lambda s: "handle_error" if s.get("error") else "train"
        )
        graph.add_conditional_edges(
            "train", lambda s: "handle_error" if s.get("error") else "execute"
        )
        graph.add_edge("execute", "evaluate")
        graph.add_edge("evaluate", END)
        graph.add_edge("handle_error", END)

        return graph.compile(checkpointer=MemorySaver())

    async def _node_classify(self, state: AgentForgeState) -> dict:
        print(f"\n[1/4] Classifying job...")
        try:
            raw = state["messages"][-1].content if state["messages"] else ""
            tenant_id = state.get("tenant_id", "default")
            job = await self.meta_agent.classify(str(raw), tenant_id)
            print(f"      Type: {job.job_type} | Domain: {job.domain}")
            return {"job": job.model_dump(mode="json"), "status": "researching"}
        except Exception as e:
            logger.error(f"Classify node error: {e}")
            return {"error": str(e), "status": "failed"}

    async def _node_train(self, state: AgentForgeState) -> dict:
        print(f"\n[2/4] Training agent...")
        try:
            from agentforge.schemas.job import JobDefinition
            job = JobDefinition(**state["job"])
            session = await self.trainer.train(job)
            for line in session.training_log:
                print(f"  {line}")
            return {
                "training_session": {
                    "job_id": session.job_id,
                    "research": session.research.model_dump(mode="json"),
                    "context_pack": session.context_pack.model_dump(mode="json"),
                    "training_log": session.training_log,
                    "duration_seconds": session.duration_seconds,
                },
                "status": "running",
            }
        except Exception as e:
            logger.error(f"Train node error: {e}")
            return {"error": str(e), "status": "failed"}

    async def _node_execute(self, state: AgentForgeState) -> dict:
        print(f"\n[3/4] Executing job...")
        try:
            from agentforge.schemas.context_pack import ContextPack
            from agentforge.schemas.job import JobDefinition

            job = JobDefinition(**state["job"])
            context_pack = ContextPack(**state["training_session"]["context_pack"])
            result = await self.worker.execute(job, context_pack, self.tool_registry)
            print(f"      Tools used: {result.tools_called}")
            print(f"      Success: {result.success}")
            return {
                "worker_result": result.model_dump(mode="json"),
                "status": "evaluating",
            }
        except Exception as e:
            logger.error(f"Execute node error: {e}")
            return {"error": str(e), "status": "failed"}

    async def _node_evaluate(self, state: AgentForgeState) -> dict:
        print(f"\n[4/4] Evaluating result...")
        try:
            from agentforge.core.worker import WorkerResult
            from agentforge.schemas.job import JobDefinition
            from agentforge.schemas.research import ResearchResult

            job = JobDefinition(**state["job"])
            worker_result = WorkerResult(**state["worker_result"])
            research = ResearchResult(**state["training_session"]["research"])
            eval_result = await self.evaluator.evaluate(job, worker_result, research)
            print(f"      Quality score: {eval_result.quality_score:.0%}")
            status = "success" if eval_result.passed else "failed"

            # Store experience in episodic memory
            if self.episodic_memory:
                try:
                    from agentforge.memory.episodic import ExperienceRecord
                    research_data = state["training_session"]["research"]
                    lessons = await self.episodic_memory.extract_lessons(job, worker_result, eval_result, self.llm_fast)
                    tools_used = worker_result.tools_called
                    approach = research_data.get("expert_approach", [""])
                    record = ExperienceRecord(
                        job_type=job.job_type,
                        domain=job.domain,
                        tenant_id=job.tenant_id,
                        job_description_summary=job.description[:200],
                        research_queries_used=research_data.get("search_queries_used", []),
                        skills_discovered=research_data.get("required_skills", []),
                        tool_types_needed=research_data.get("required_tool_types", []),
                        tools_that_worked=tools_used,
                        quality_score=eval_result.quality_score,
                        research_confidence=research_data.get("confidence", 0.0),
                        lessons=lessons,
                        approach_summary=f"Used {tools_used} with approach: {approach[0] if approach else 'standard'}",
                        duration_seconds=state["training_session"].get("duration_seconds", 0.0),
                    )
                    await self.episodic_memory.remember(record)
                    print("      💾 Experience stored for future jobs")
                except Exception as e:
                    logger.warning(f"Failed to store experience: {e}")

            return {"evaluation": eval_result.model_dump(), "status": status}
        except Exception as e:
            logger.error(f"Evaluate node error: {e}")
            return {"evaluation": {"quality_score": 0, "passed": False, "feedback": str(e)}, "status": "failed"}

    async def _node_handle_error(self, state: AgentForgeState) -> dict:
        error = state.get("error", "Unknown error")
        logger.error(f"Pipeline error: {error}")
        print(f"\n❌ Pipeline error: {error}")
        return {"status": "failed"}

    async def run_job(
        self, raw_input: str, tenant_id: str = "default", config: Optional[dict] = None
    ) -> dict:
        """Run a job through the full pipeline. Returns the final state dict."""
        from langchain_core.messages import HumanMessage

        if config is None:
            config = {"configurable": {"thread_id": f"job_{tenant_id}"}}

        print(f"\n{'='*60}")
        print(f"AgentForge — Processing: {raw_input[:80]}")
        print(f"{'='*60}")

        initial_state: AgentForgeState = {
            "messages": [HumanMessage(content=raw_input)],
            "job": None,
            "training_session": None,
            "worker_result": None,
            "evaluation": None,
            "error": None,
            "status": "pending",
            "tenant_id": tenant_id,
        }

        final_state = await self._graph.ainvoke(initial_state, config=config)
        print(f"\n{'='*60}")
        print(f"Status: {final_state.get('status', 'unknown')}")
        return final_state
