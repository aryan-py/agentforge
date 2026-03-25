"""Tool registry — maps research-discovered tool_type strings to callable tools."""

import logging
from typing import Any, List, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class ToolRegistryEntry(BaseModel):
    """Metadata for a registered tool, including the tool_types it covers.

    tool_types MUST align with the strings that ResearchResultParser extracts
    (e.g. "web search", "calculator"). This is the bridge between research
    findings and actual callable tools.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_id: str  # slug: "web_search", "calculator"
    name: str  # human-readable name
    description: str  # 2-3 sentences: what it does and when to use it
    tool_types: List[str]  # CRITICAL: must match what ResearchResultParser extracts
    capability_tags: List[str] = []  # additional searchable tags
    source: str = "builtin"  # "builtin" or "mcp://server-name"
    requires_auth: bool = False
    allowed_tenants: List[str] = ["*"]
    cost_tier: str = "low"  # "free", "low", "medium", "high"
    langchain_tool: Optional[Any] = None  # the actual callable BaseTool


class MatchResult(BaseModel):
    """Result of matching research requirements against the tool registry."""

    matched: List[ToolRegistryEntry]  # tools found for requirements
    unmatched: List[str]  # requirements with no tool found
    coverage: float  # matched_requirements / total_requirements

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ToolRegistry:
    """Registry that maps ResearchResult.required_tool_types to callable LangChain tools.

    The core bridge between what the trainer discovers via web research and what
    tools the worker agent actually receives. Matching is exact-first, then fuzzy
    (word overlap) to handle slight phrasing differences.
    """

    def __init__(self):
        self._entries: dict[str, ToolRegistryEntry] = {}

    def register(self, entry: ToolRegistryEntry) -> None:
        """Add a tool to the registry."""
        self._entries[entry.tool_id] = entry
        logger.debug(f"Registered tool: {entry.tool_id} (types: {entry.tool_types})")

    def find_for_requirements(
        self, required_tool_types: List[str], tenant_id: str = "*"
    ) -> MatchResult:
        """Map a list of required_tool_types to registered tools.

        Matching strategy:
        1. Exact string match against entry.tool_types
        2. Fuzzy match: check if any word in the requirement appears in any tool_type
        Returns a MatchResult with matched tools and unmatched requirements.
        """
        matched_entries: List[ToolRegistryEntry] = []
        matched_tool_ids: set[str] = set()
        unmatched: List[str] = []
        matched_requirements = 0

        for req in required_tool_types:
            req_lower = req.lower().strip()
            found: Optional[ToolRegistryEntry] = None

            # Pass 1: exact match
            for entry in self._entries.values():
                if not self._tenant_allowed(entry, tenant_id):
                    continue
                if req_lower in [t.lower() for t in entry.tool_types]:
                    found = entry
                    break

            # Pass 2: fuzzy (word overlap)
            if not found:
                req_words = set(req_lower.split())
                best_score = 0
                for entry in self._entries.values():
                    if not self._tenant_allowed(entry, tenant_id):
                        continue
                    for tool_type in entry.tool_types:
                        type_words = set(tool_type.lower().split())
                        overlap = len(req_words & type_words)
                        if overlap > best_score:
                            best_score = overlap
                            found = entry

            if found:
                matched_requirements += 1
                if found.tool_id not in matched_tool_ids:
                    matched_tool_ids.add(found.tool_id)
                    matched_entries.append(found)
            else:
                unmatched.append(req)

        total = len(required_tool_types)
        coverage = matched_requirements / total if total > 0 else 0.0

        return MatchResult(matched=matched_entries, unmatched=unmatched, coverage=coverage)

    def get_tool(self, tool_id: str) -> Optional[BaseTool]:
        """Return the callable LangChain BaseTool for a given tool_id."""
        entry = self._entries.get(tool_id)
        if entry and entry.langchain_tool:
            return entry.langchain_tool
        return None

    def resolve_tools(self, tool_ids: List[str]) -> List[BaseTool]:
        """Batch-resolve a list of tool_ids to LangChain BaseTool instances."""
        tools = []
        for tool_id in tool_ids:
            tool = self.get_tool(tool_id)
            if tool:
                tools.append(tool)
            else:
                logger.warning(f"Could not resolve tool: {tool_id}")
        return tools

    def list_all_tool_types(self) -> List[str]:
        """Return all unique tool_type strings across all registered tools."""
        types: set[str] = set()
        for entry in self._entries.values():
            types.update(entry.tool_types)
        return sorted(types)

    def _tenant_allowed(self, entry: ToolRegistryEntry, tenant_id: str) -> bool:
        return "*" in entry.allowed_tenants or tenant_id in entry.allowed_tenants

    async def load_mcp_tools(self, mcp_configs: List[dict], llm=None) -> int:
        """Load tools from MCP servers and register them with inferred tool_types.

        mcp_configs format: [{"name": "server_name", "transport": "stdio", "command": "npx", "args": [...]}]
        Uses an LLM to infer tool_types from each tool's name and description.
        """
        import json as _json
        from langchain_mcp_adapters.client import MultiServerMCPClient

        if not mcp_configs:
            return 0

        loaded = 0
        try:
            # Build config dict for MultiServerMCPClient
            client_config = {}
            for cfg in mcp_configs:
                name = cfg.get("name", f"server_{loaded}")
                client_config[name] = {
                    "transport": cfg.get("transport", "stdio"),
                    "command": cfg.get("command", "npx"),
                    "args": cfg.get("args", []),
                }

            async with MultiServerMCPClient(client_config) as client:
                tools = client.get_tools()
                for tool in tools:
                    server_name = getattr(tool, "_server_name", "mcp")
                    # Infer tool_types via LLM if available
                    tool_types = [tool.name.replace("_", " ")]
                    if llm:
                        try:
                            from langchain_core.messages import HumanMessage, SystemMessage
                            resp = await llm.ainvoke([
                                SystemMessage(content="List 3-5 tool_type strings describing what this tool does. Return JSON array only."),
                                HumanMessage(content=f"Tool: {tool.name} — {tool.description[:200]}"),
                            ])
                            raw = resp.content.strip().strip("```json").strip("```").strip()
                            tool_types = _json.loads(raw)
                        except Exception:
                            pass

                    entry = ToolRegistryEntry(
                        tool_id=f"mcp_{tool.name}",
                        name=tool.name,
                        description=tool.description or f"MCP tool: {tool.name}",
                        tool_types=tool_types,
                        source=f"mcp://{server_name}",
                        langchain_tool=tool,
                    )
                    self.register(entry)
                    loaded += 1

            logger.info(f"MCP tools loaded: {loaded} tools from {len(mcp_configs)} servers")
        except Exception as e:
            logger.warning(f"load_mcp_tools() failed: {e}")

        return loaded
