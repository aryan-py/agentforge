"""Langfuse observability — traces every LLM call, tool call, and pipeline node."""

import logging
from contextlib import contextmanager
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ObservabilityManager:
    """Singleton wrapper around Langfuse for end-to-end pipeline tracing.

    Instruments each pipeline node as a Langfuse span, with sub-spans for
    every LLM call and tool execution. Tracks token counts and costs.
    Gracefully no-ops when Langfuse is not configured.
    """

    def __init__(
        self,
        public_key: str = "",
        secret_key: str = "",
        host: str = "https://cloud.langfuse.com",
    ):
        self._enabled = bool(public_key and secret_key)
        self._client = None
        if self._enabled:
            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                logger.info("Langfuse observability initialized")
            except Exception as e:
                logger.warning(f"Langfuse init failed: {e}")
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_trace(
        self,
        name: str,
        job_id: str,
        tenant_id: str,
        job_type: str = "",
        domain: str = "",
        tags: Optional[list] = None,
    ):
        """Start a new Langfuse trace for a job. Returns trace handle or None."""
        if not self._enabled or self._client is None:
            return None
        try:
            return self._client.trace(
                name=name,
                id=job_id,
                user_id=tenant_id,
                metadata={
                    "job_type": job_type,
                    "domain": domain,
                    "tenant_id": tenant_id,
                },
                tags=tags or [],
            )
        except Exception as e:
            logger.debug(f"start_trace failed: {e}")
            return None

    def span(self, trace, name: str, metadata: Optional[dict] = None):
        """Create a child span on an existing trace. Returns span or None."""
        if not self._enabled or trace is None:
            return None
        try:
            return trace.span(name=name, metadata=metadata or {})
        except Exception as e:
            logger.debug(f"span() failed: {e}")
            return None

    def end_span(self, span, output: Any = None, level: str = "DEFAULT") -> None:
        """End a span with optional output."""
        if span is None:
            return
        try:
            span.end(output=str(output)[:500] if output else None, level=level)
        except Exception as e:
            logger.debug(f"end_span() failed: {e}")

    def log_generation(
        self,
        trace,
        name: str,
        model: str,
        prompt: str,
        completion: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Log an LLM generation with token counts and cost."""
        if not self._enabled or trace is None:
            return
        try:
            trace.generation(
                name=name,
                model=model,
                input=prompt[:500],
                output=completion[:500],
                usage={"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
                metadata={"cost_usd": cost_usd},
            )
        except Exception as e:
            logger.debug(f"log_generation() failed: {e}")

    def score(self, trace, name: str, value: float, comment: str = "") -> None:
        """Record a quality score on a trace."""
        if not self._enabled or trace is None:
            return
        try:
            self._client.score(
                trace_id=trace.id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as e:
            logger.debug(f"score() failed: {e}")

    def flush(self) -> None:
        """Flush any pending events to Langfuse."""
        if self._enabled and self._client:
            try:
                self._client.flush()
            except Exception:
                pass
