"""Job submission and status API routes."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from agentforge.schemas.job import JobResult

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobRequest(BaseModel):
    description: str
    tenant_id: str = "default"


@router.post("", response_model=dict)
async def submit_job(request: Request, body: JobRequest):
    """Submit a job to the AgentForge pipeline."""
    pipeline = request.app.state.pipeline
    try:
        state = await pipeline.run_job(body.description, body.tenant_id)

        job = state.get("job") or {}
        worker = state.get("worker_result") or {}
        evaluation = state.get("evaluation") or {}
        training = state.get("training_session") or {}

        return {
            "job_id": job.get("job_id", ""),
            "status": state.get("status", "unknown"),
            "output": worker.get("output", ""),
            "quality_score": evaluation.get("quality_score"),
            "error": state.get("error"),
            "tools_used": worker.get("tools_called", []),
            "research_queries": training.get("research", {}).get("search_queries_used", []),
            "skills_discovered": training.get("research", {}).get("required_skills", []),
            "cost_usd": 0.0,
            "duration_seconds": training.get("duration_seconds", 0.0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
