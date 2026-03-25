"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings

_ENV_FILE = Path(__file__).parent.parent.parent.parent / ".env"


class Settings(BaseSettings):
    """All configuration for AgentForge loaded from environment / .env file.

    Each field maps to an environment variable of the same name (uppercase).
    """

    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    QDRANT_URL: str = "http://localhost:6333"
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    DEFAULT_MODEL: str = "claude-sonnet-4-6"
    PLANNER_MODEL: str = "claude-opus-4-6"
    ROUTER_MODEL: str = "gpt-4o-mini"
    RESEARCH_CACHE_TTL_DAYS: int = 7
    MAX_SEARCH_QUERIES_PER_JOB: int = 5
    MAX_PAGES_TO_FETCH: int = 3
    ENABLE_MCP_DISCOVERY: bool = True
    MCP_SERVERS: str = "[]"  # JSON array of MCP server configs
    MAX_CONCURRENT_JOBS: int = 10
    TEMPORAL_HOST: str = "localhost:7233"

    model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()

# Propagate keys into os.environ so third-party clients (openai, anthropic) pick them up
import os as _os  # noqa: E402
if settings.OPENAI_API_KEY:
    _os.environ.setdefault("OPENAI_API_KEY", settings.OPENAI_API_KEY)
if settings.ANTHROPIC_API_KEY:
    _os.environ.setdefault("ANTHROPIC_API_KEY", settings.ANTHROPIC_API_KEY)
