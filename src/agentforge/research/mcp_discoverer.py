"""Discovers MCP servers that can fill tool gaps identified during research."""

import json
import logging
import subprocess
from pathlib import Path
from typing import List

from pydantic import BaseModel

from agentforge.research.web_searcher import WebSearcher

logger = logging.getLogger(__name__)

_SUGGESTIONS_LOG = Path("logs/mcp_suggestions.jsonl")


class MCPServerSuggestion(BaseModel):
    """A suggested MCP server package that could fill a tool capability gap."""

    package_name: str  # e.g. "@modelcontextprotocol/server-brave-search"
    transport: str  # "stdio" or "http"
    capability_covered: str  # what gap this fills
    install_command: str  # e.g. "npx -y @modelcontextprotocol/server-brave-search"
    notes: str  # any required API keys or config


class MCPDiscoverer:
    """Searches the web for MCP server packages to fill tool capability gaps.

    When the TrainerAgent finds that no built-in tool covers a required_tool_type,
    MCPDiscoverer suggests installable MCP servers and logs them for admin review.
    """

    def __init__(self):
        self.searcher = WebSearcher(max_results_per_query=5, fetch_pages=False)
        _SUGGESTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)

    async def find_servers_for_capability(
        self, capability: str
    ) -> List[MCPServerSuggestion]:
        """Search for MCP servers that provide the given capability."""
        query = f"modelcontextprotocol {capability} mcp server npm"
        results = await self.searcher.search(query)

        suggestions: List[MCPServerSuggestion] = []
        for r in results:
            text = f"{r.title} {r.snippet}".lower()
            # Look for @modelcontextprotocol/* package names
            import re

            packages = re.findall(r"@[\w-]+/[\w-]+", text)
            for pkg in packages:
                if "modelcontextprotocol" in pkg or "mcp" in pkg:
                    suggestions.append(
                        MCPServerSuggestion(
                            package_name=pkg,
                            transport="stdio",
                            capability_covered=capability,
                            install_command=f"npx -y {pkg}",
                            notes=f"Found via search for '{capability}'. Check package docs for required env vars.",
                        )
                    )

        # Deduplicate by package name
        seen = set()
        unique = []
        for s in suggestions:
            if s.package_name not in seen:
                seen.add(s.package_name)
                unique.append(s)

        return unique[:3]  # top 3 suggestions

    async def is_installed(self, package_name: str) -> bool:
        """Check if an MCP server package is already installed via npx."""
        try:
            result = subprocess.run(
                ["npx", "--yes", "--dry-run", package_name],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    async def save_suggestions(
        self, suggestions: List[MCPServerSuggestion], job_id: str
    ) -> None:
        """Append suggestions to the JSONL log for admin review."""
        if not suggestions:
            return
        with _SUGGESTIONS_LOG.open("a") as f:
            for s in suggestions:
                entry = {"job_id": job_id, **s.model_dump()}
                f.write(json.dumps(entry) + "\n")
