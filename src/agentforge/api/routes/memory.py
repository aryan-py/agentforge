"""Memory statistics API route."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/stats")
async def memory_stats(request: Request, tenant_id: str = "*"):
    """Return episodic memory and research cache statistics."""
    try:
        cache = request.app.state.research_cache
        cache_stats = await cache.stats()

        stats = {
            "research_cache": cache_stats,
            "episodic_memory": {"total_experiences": 0},
        }

        if hasattr(request.app.state, "episodic_memory") and request.app.state.episodic_memory:
            mem_stats = await request.app.state.episodic_memory.stats(tenant_id)
            stats["episodic_memory"] = mem_stats

        return stats
    except Exception as e:
        return {"error": str(e)}
