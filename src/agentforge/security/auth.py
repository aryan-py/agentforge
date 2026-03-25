"""Tenant registry and RBAC middleware for multi-tenant AgentForge deployments."""

import logging
from typing import Dict, List, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TenantConfig(BaseModel):
    """Per-tenant configuration for access control and cost management."""

    tenant_id: str
    name: str
    allowed_domains: List[str] = ["*"]  # "*" = all domains
    allowed_tool_types: List[str] = ["*"]
    max_cost_per_day_usd: float = 50.0
    requires_approval_for: List[str] = []  # job_types requiring HITL
    isolation_mode: str = "shared"  # "shared" or "namespace"
    active: bool = True


class TenantRegistry:
    """Registry of all known tenants and their configurations.

    Used by RBAC middleware to validate tenant IDs and enforce per-tenant
    access controls on domains, tool types, and cost limits.
    """

    def __init__(self):
        self._tenants: Dict[str, TenantConfig] = {}
        # Register default tenant
        self.register(
            TenantConfig(
                tenant_id="default",
                name="Default Tenant",
                allowed_domains=["*"],
                max_cost_per_day_usd=10.0,
            )
        )

    def register(self, config: TenantConfig) -> None:
        self._tenants[config.tenant_id] = config
        logger.info(f"Registered tenant: {config.tenant_id}")

    def get(self, tenant_id: str) -> Optional[TenantConfig]:
        return self._tenants.get(tenant_id)

    def is_valid(self, tenant_id: str) -> bool:
        t = self._tenants.get(tenant_id)
        return t is not None and t.active

    def requires_approval(self, tenant_id: str, job_type: str) -> bool:
        t = self._tenants.get(tenant_id)
        if t is None:
            return False
        return job_type in t.requires_approval_for

    def list_tenants(self) -> List[TenantConfig]:
        return list(self._tenants.values())


class RBACMiddleware(BaseHTTPMiddleware):
    """Validates X-Tenant-ID header on every request (except /health).

    Sets request.state.tenant_id for use by route handlers.
    Falls back to "default" tenant when no header is present.
    """

    def __init__(self, app, registry: TenantRegistry, strict: bool = False):
        super().__init__(app)
        self.registry = registry
        self.strict = strict  # if True, reject requests without header

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health and docs routes
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        tenant_id = request.headers.get("X-Tenant-ID", "default")

        if self.strict and tenant_id == "default":
            return JSONResponse(
                status_code=401,
                content={"error": "X-Tenant-ID header required"},
            )

        if not self.registry.is_valid(tenant_id):
            return JSONResponse(
                status_code=403,
                content={"error": f"Unknown or inactive tenant: {tenant_id}"},
            )

        request.state.tenant_id = tenant_id
        return await call_next(request)
