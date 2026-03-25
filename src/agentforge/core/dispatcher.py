"""JobDispatcher — submits jobs to Temporal and tracks workflow state."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

TASK_QUEUE = "agentforge-tasks"


class JobDispatcher:
    """Submits jobs to Temporal as durable workflows and tracks their state.

    Used by the FastAPI routes in Phase 3 to submit long-running jobs
    that must survive process crashes.
    """

    def __init__(self, temporal_host: str = "localhost:7233"):
        self.temporal_host = temporal_host
        self._client = None

    async def connect(self) -> None:
        """Connect to the Temporal server."""
        from temporalio.client import Client
        self._client = await Client.connect(self.temporal_host)
        logger.info(f"JobDispatcher connected to Temporal at {self.temporal_host}")

    async def submit(self, raw_input: str, tenant_id: str) -> str:
        """Submit a job as a Temporal workflow. Returns workflow_id."""
        if self._client is None:
            raise RuntimeError("JobDispatcher not connected. Call connect() first.")

        from agentforge.core.workflows import AgentForgeWorkflow, WorkflowParams
        import uuid

        workflow_id = f"job-{tenant_id}-{uuid.uuid4().hex[:8]}"
        params = WorkflowParams(raw_input=raw_input, tenant_id=tenant_id)

        handle = await self._client.start_workflow(
            AgentForgeWorkflow.run,
            params,
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
        logger.info(f"Submitted workflow: {workflow_id}")
        return workflow_id

    async def get_status(self, workflow_id: str) -> dict:
        """Get the current state of a running workflow."""
        if self._client is None:
            raise RuntimeError("Not connected.")
        handle = self._client.get_workflow_handle(workflow_id)
        try:
            status = await handle.query(AgentForgeWorkflow.status)
            desc = await handle.describe()
            return {
                "workflow_id": workflow_id,
                "status": status,
                "workflow_status": str(desc.status),
            }
        except Exception as e:
            return {"workflow_id": workflow_id, "error": str(e)}

    async def approve(self, workflow_id: str, reviewer_id: str) -> None:
        """Send an approval signal to a paused workflow."""
        if self._client is None:
            raise RuntimeError("Not connected.")
        handle = self._client.get_workflow_handle(workflow_id)
        await handle.signal(AgentForgeWorkflow.approve, reviewer_id)

    async def deny(self, workflow_id: str, reviewer_id: str, reason: str = "") -> None:
        """Send a denial signal to a paused workflow."""
        if self._client is None:
            raise RuntimeError("Not connected.")
        handle = self._client.get_workflow_handle(workflow_id)
        await handle.signal(AgentForgeWorkflow.deny, reviewer_id, reason)
