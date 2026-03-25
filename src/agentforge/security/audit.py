"""Immutable append-only audit log for all AgentForge events."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_AUDIT_DIR = Path("logs/audit")


class AuditLogger:
    """Writes immutable JSONL audit entries for compliance and incident response.

    Each event type gets its own file: jobs.jsonl, tool_calls.jsonl, approvals.jsonl.
    Files are append-only — entries are never deleted or modified.
    """

    def __init__(self, audit_dir: Path = _AUDIT_DIR):
        self.audit_dir = audit_dir
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, filename: str, entry: Dict[str, Any]) -> None:
        path = self.audit_dir / filename
        entry["_logged_at"] = datetime.utcnow().isoformat()
        try:
            with path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Audit log write failed ({filename}): {e}")

    def log_job_submitted(
        self,
        job_id: str,
        tenant_id: str,
        description: str,
        requester_id: str = "api",
    ) -> None:
        self._write(
            "jobs.jsonl",
            {
                "event": "job_submitted",
                "job_id": job_id,
                "tenant_id": tenant_id,
                "description": description[:200],
                "requester_id": requester_id,
            },
        )

    def log_job_completed(
        self,
        job_id: str,
        tenant_id: str,
        status: str,
        quality_score: Optional[float],
        cost_usd: float,
        duration_seconds: float,
    ) -> None:
        self._write(
            "jobs.jsonl",
            {
                "event": "job_completed",
                "job_id": job_id,
                "tenant_id": tenant_id,
                "status": status,
                "quality_score": quality_score,
                "cost_usd": cost_usd,
                "duration_seconds": duration_seconds,
            },
        )

    def log_tool_call(
        self,
        job_id: str,
        tenant_id: str,
        tool_id: str,
        input_preview: str,
    ) -> None:
        self._write(
            "tool_calls.jsonl",
            {
                "event": "tool_called",
                "job_id": job_id,
                "tenant_id": tenant_id,
                "tool_id": tool_id,
                "input_preview": input_preview[:200],
            },
        )

    def log_approval_event(
        self,
        job_id: str,
        tenant_id: str,
        event_type: str,  # "requested", "approved", "denied"
        reviewer_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        self._write(
            "approvals.jsonl",
            {
                "event": f"approval_{event_type}",
                "job_id": job_id,
                "tenant_id": tenant_id,
                "reviewer_id": reviewer_id,
                "reason": reason,
            },
        )
