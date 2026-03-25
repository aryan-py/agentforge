"""Temporal worker process — runs activities for AgentForge workflows."""

import asyncio
import logging

from temporalio.client import Client
from temporalio.worker import Worker

from agentforge.core.activities import (
    classify_job_activity,
    evaluate_result_activity,
    execute_job_activity,
    request_approval_activity,
    train_agent_activity,
)
from agentforge.core.workflows import AgentForgeWorkflow

logger = logging.getLogger(__name__)

TASK_QUEUE = "agentforge-tasks"


async def run_worker(temporal_host: str = "localhost:7233") -> None:
    """Connect to Temporal and start processing workflow tasks."""
    client = await Client.connect(temporal_host)
    logger.info(f"Connected to Temporal at {temporal_host}")

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AgentForgeWorkflow],
        activities=[
            classify_job_activity,
            train_agent_activity,
            request_approval_activity,
            execute_job_activity,
            evaluate_result_activity,
        ],
    )

    logger.info(f"Worker started on task queue: {TASK_QUEUE}")
    await worker.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())
