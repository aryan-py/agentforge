"""Temporal workflow definitions for durable AgentForge job execution."""

from datetime import timedelta
from typing import Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

from agentforge.core.activities import (
    classify_job_activity,
    evaluate_result_activity,
    execute_job_activity,
    request_approval_activity,
    train_agent_activity,
)


class WorkflowParams:
    """Input parameters for AgentForgeWorkflow."""

    def __init__(self, raw_input: str, tenant_id: str):
        self.raw_input = raw_input
        self.tenant_id = tenant_id


class WorkflowResult:
    """Output of AgentForgeWorkflow.run()."""

    def __init__(
        self,
        status: str,
        result: Optional[dict] = None,
        evaluation: Optional[dict] = None,
    ):
        self.status = status
        self.result = result or {}
        self.evaluation = evaluation or {}


@workflow.defn
class AgentForgeWorkflow:
    """Durable Temporal workflow wrapping the full AgentForge pipeline.

    Provides crash recovery, retries, and human-in-the-loop approval gates
    for high-risk jobs. Each stage runs as a separate Activity with its own
    timeout and retry policy.
    """

    def __init__(self):
        self._approved = False
        self._denied = False
        self._denial_reason = ""

    @workflow.run
    async def run(self, params: WorkflowParams) -> WorkflowResult:
        """Execute the full classify → train → [approve] → execute → evaluate pipeline."""

        # Activity 1: Classify job
        job_dict = await workflow.execute_activity(
            classify_job_activity,
            args=[params.raw_input, params.tenant_id],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Activity 2: Train agent (web research — may take 30-60s)
        training_dict = await workflow.execute_activity(
            train_agent_activity,
            args=[job_dict],
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        # Gate: pause for human approval if high-risk
        if training_dict.get("context_pack", {}).get("risk_level") == "high":
            await workflow.execute_activity(
                request_approval_activity,
                args=[job_dict, training_dict],
                start_to_close_timeout=timedelta(seconds=10),
            )
            approved = await workflow.wait_condition(
                lambda: self._approved or self._denied,
                timeout=timedelta(hours=24),
            )
            if self._denied or not approved:
                return WorkflowResult(
                    status="denied",
                    result={"reason": self._denial_reason},
                )

        # Activity 3: Execute job
        result_dict = await workflow.execute_activity(
            execute_job_activity,
            args=[job_dict, training_dict],
            start_to_close_timeout=timedelta(
                seconds=job_dict.get("timeout_seconds", 300)
            ),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )

        # Activity 4: Evaluate result
        eval_dict = await workflow.execute_activity(
            evaluate_result_activity,
            args=[job_dict, result_dict, training_dict],
            start_to_close_timeout=timedelta(seconds=30),
        )

        return WorkflowResult(
            status="success", result=result_dict, evaluation=eval_dict
        )

    @workflow.signal
    def approve(self, reviewer_id: str) -> None:
        """Signal to approve a high-risk job execution."""
        self._approved = True

    @workflow.signal
    def deny(self, reviewer_id: str, reason: str = "") -> None:
        """Signal to deny a high-risk job execution."""
        self._denied = True
        self._denial_reason = reason

    @workflow.query
    def status(self) -> str:
        """Query the current approval status."""
        if self._approved:
            return "approved"
        if self._denied:
            return "denied"
        return "pending_approval"
