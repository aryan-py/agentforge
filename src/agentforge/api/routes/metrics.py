"""Metrics API route — last-7-days job stats per tenant."""

import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/metrics", tags=["metrics"])

_AUDIT_DIR = Path("logs/audit")


@router.get("")
async def get_metrics(tenant_id: str = "*", days: int = 7):
    """Return job performance summary for the last N days."""
    jobs_file = _AUDIT_DIR / "jobs.jsonl"
    if not jobs_file.exists():
        return {"tenant_id": tenant_id, "days": days, "total_jobs": 0}

    from datetime import datetime, timedelta

    cutoff = datetime.utcnow() - timedelta(days=days)
    submissions = []
    completions = []

    try:
        for line in jobs_file.read_text().splitlines():
            try:
                entry = json.loads(line)
                logged_at = datetime.fromisoformat(entry.get("_logged_at", "2000-01-01"))
                if logged_at < cutoff:
                    continue
                if tenant_id != "*" and entry.get("tenant_id") != tenant_id:
                    continue
                if entry.get("event") == "job_submitted":
                    submissions.append(entry)
                elif entry.get("event") == "job_completed":
                    completions.append(entry)
            except Exception:
                pass
    except Exception:
        pass

    quality_scores = [c["quality_score"] for c in completions if c.get("quality_score") is not None]
    total_cost = sum(c.get("cost_usd", 0.0) for c in completions)
    durations = [c.get("duration_seconds", 0.0) for c in completions]

    return {
        "tenant_id": tenant_id,
        "days": days,
        "total_jobs_submitted": len(submissions),
        "total_jobs_completed": len(completions),
        "avg_quality_score": round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else None,
        "avg_duration_seconds": round(sum(durations) / len(durations), 1) if durations else None,
        "total_cost_usd": round(total_cost, 4),
    }
