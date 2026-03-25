"""File-based cache for ResearchResult objects, keyed by job_type + domain."""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from agentforge.schemas.research import ResearchResult

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(".cache/research")


class ResearchCache:
    """Stores ResearchResult objects as JSON files to avoid redundant web searches.

    Cache key: sha256(f"{job_type}:{domain}")[:16]
    TTL: configurable via RESEARCH_CACHE_TTL_DAYS (default 7 days).
    """

    def __init__(self, cache_dir: Path = _CACHE_DIR, ttl_days: int = 7):
        self.cache_dir = cache_dir
        self.ttl = timedelta(days=ttl_days)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, job_type: str, domain: str) -> str:
        return hashlib.sha256(f"{job_type}:{domain}".encode()).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    async def get(self, job_type: str, domain: str) -> Optional[ResearchResult]:
        """Return a cached ResearchResult if it exists and is not expired."""
        path = self._path(self._key(job_type, domain))
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            cached_at = datetime.fromisoformat(data["_cached_at"])
            if datetime.utcnow() - cached_at > self.ttl:
                logger.debug(f"Cache expired for {job_type}/{domain}")
                return None
            del data["_cached_at"]
            return ResearchResult(**data)
        except Exception as e:
            logger.warning(f"Cache read error for {job_type}/{domain}: {e}")
            return None

    async def set(self, job_type: str, domain: str, result: ResearchResult) -> None:
        """Write a ResearchResult to the cache with a timestamp."""
        path = self._path(self._key(job_type, domain))
        try:
            data = json.loads(result.model_dump_json())
            data["_cached_at"] = datetime.utcnow().isoformat()
            path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.warning(f"Cache write error for {job_type}/{domain}: {e}")

    async def invalidate(self, job_type: str, domain: str) -> None:
        """Delete the cache entry for a job_type/domain combination."""
        path = self._path(self._key(job_type, domain))
        if path.exists():
            path.unlink()

    async def stats(self) -> dict:
        """Return statistics about the current cache state."""
        files = list(self.cache_dir.glob("*.json"))
        if not files:
            return {"total_cached": 0, "oldest_entry": None, "newest_entry": None, "cache_size_kb": 0}

        timestamps = []
        total_bytes = 0
        for f in files:
            try:
                data = json.loads(f.read_text())
                timestamps.append(datetime.fromisoformat(data["_cached_at"]))
                total_bytes += f.stat().st_size
            except Exception:
                pass

        return {
            "total_cached": len(files),
            "oldest_entry": min(timestamps).isoformat() if timestamps else None,
            "newest_entry": max(timestamps).isoformat() if timestamps else None,
            "cache_size_kb": round(total_bytes / 1024, 2),
        }
