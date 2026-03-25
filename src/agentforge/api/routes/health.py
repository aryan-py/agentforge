"""Health check route."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    """Return system health and registry stats."""
    try:
        registry = request.app.state.tool_registry
        cache = request.app.state.research_cache
        cache_stats = await cache.stats()
        tool_types = registry.list_all_tool_types()
        return {
            "status": "ok",
            "tools_registered": len(tool_types),
            "cache_entries": cache_stats.get("total_cached", 0),
        }
    except Exception:
        return {"status": "ok", "tools_registered": 0, "cache_entries": 0}
