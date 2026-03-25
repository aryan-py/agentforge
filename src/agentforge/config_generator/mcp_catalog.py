"""Curated catalog of known MCP servers with metadata for config generation."""

from typing import List, Optional
from pydantic import BaseModel


class MCPServerSpec(BaseModel):
    """Full metadata for a known MCP server."""

    id: str
    name: str
    package: str                      # npm package name
    command: str = "npx"
    args: List[str]                   # full args including package
    description: str                  # what it does, when to use it
    tool_types: List[str]             # matches ResearchResult.required_tool_types
    requires_key: bool = False
    key_name: Optional[str] = None    # env var name, e.g. "BRAVE_API_KEY"
    key_url: Optional[str] = None     # where to get the key
    key_instructions: Optional[str] = None
    extra_placeholders: dict = {}     # other args the user must fill in
    python_packages: List[str] = []   # pip packages that pair with this server
    free_tier_available: bool = True


# ── Catalog ───────────────────────────────────────────────────────────────────
CATALOG: List[MCPServerSpec] = [
    MCPServerSpec(
        id="brave_search",
        name="Brave Search",
        package="@modelcontextprotocol/server-brave-search",
        args=["-y", "@modelcontextprotocol/server-brave-search"],
        description="Real-time web search via the Brave Search API. Use for current events, fact lookup, and research.",
        tool_types=["web search", "real-time information", "fact lookup", "current events", "research", "news", "internet search"],
        requires_key=True,
        key_name="BRAVE_API_KEY",
        key_url="https://brave.com/search/api/",
        key_instructions="Sign up at brave.com/search/api — free tier includes 2,000 queries/month.",
        free_tier_available=True,
    ),
    MCPServerSpec(
        id="fetch",
        name="Fetch (URL Reader)",
        package="@modelcontextprotocol/server-fetch",
        args=["-y", "@modelcontextprotocol/server-fetch"],
        description="Fetch and read the content of any URL. No API key required.",
        tool_types=["web page reader", "URL reader", "article reader", "content extraction", "document fetcher", "http request", "link reader"],
        requires_key=False,
    ),
    MCPServerSpec(
        id="filesystem",
        name="Filesystem",
        package="@modelcontextprotocol/server-filesystem",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/YOUR/ALLOWED/DIRECTORY"],
        description="Read and write files in directories you explicitly allow. Replace the path with your project directory.",
        tool_types=["file reader", "document reader", "file writer", "report generator", "text reader", "PDF reader", "output writer", "file saver"],
        requires_key=False,
        extra_placeholders={"path": "Replace /YOUR/ALLOWED/DIRECTORY with the absolute path to your project or data folder."},
    ),
    MCPServerSpec(
        id="sqlite",
        name="SQLite",
        package="@modelcontextprotocol/server-sqlite",
        args=["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "/YOUR/DATABASE.sqlite"],
        description="Query and manage a SQLite database. Replace the db-path with your database file.",
        tool_types=["database query", "SQL", "structured data", "data retrieval", "table lookup", "spreadsheet"],
        requires_key=False,
        extra_placeholders={"db_path": "Replace /YOUR/DATABASE.sqlite with the absolute path to your SQLite file."},
    ),
    MCPServerSpec(
        id="postgres",
        name="PostgreSQL",
        package="@modelcontextprotocol/server-postgres",
        args=["-y", "@modelcontextprotocol/server-postgres", "YOUR_POSTGRES_CONNECTION_STRING"],
        description="Full PostgreSQL database access including queries, schema inspection, and writes.",
        tool_types=["database query", "SQL", "structured data", "relational data", "data retrieval"],
        requires_key=True,
        key_name="POSTGRES_CONNECTION_STRING",
        key_instructions='Replace YOUR_POSTGRES_CONNECTION_STRING with: postgresql://user:password@host:5432/dbname',
        free_tier_available=True,
    ),
    MCPServerSpec(
        id="github",
        name="GitHub",
        package="@modelcontextprotocol/server-github",
        args=["-y", "@modelcontextprotocol/server-github"],
        description="Read repos, create issues/PRs, search code, manage files on GitHub.",
        tool_types=["code review", "version control", "git", "source code", "repository", "pull request", "issue tracker"],
        requires_key=True,
        key_name="GITHUB_PERSONAL_ACCESS_TOKEN",
        key_url="https://github.com/settings/tokens",
        key_instructions="Create a token at github.com/settings/tokens. Scopes needed: repo, read:user.",
        python_packages=["PyGithub"],
    ),
    MCPServerSpec(
        id="memory",
        name="Memory (Knowledge Graph)",
        package="@modelcontextprotocol/server-memory",
        args=["-y", "@modelcontextprotocol/server-memory"],
        description="Persistent in-session memory for agents. Stores entities and relationships as a knowledge graph.",
        tool_types=["memory", "knowledge base", "persistent storage", "context storage", "entity tracking"],
        requires_key=False,
    ),
    MCPServerSpec(
        id="puppeteer",
        name="Puppeteer (Browser Automation)",
        package="@modelcontextprotocol/server-puppeteer",
        args=["-y", "@modelcontextprotocol/server-puppeteer"],
        description="Headless browser automation — navigate pages, click, fill forms, take screenshots.",
        tool_types=["web scraping", "browser automation", "screenshot", "form filling", "UI testing", "page interaction"],
        requires_key=False,
    ),
    MCPServerSpec(
        id="slack",
        name="Slack",
        package="@modelcontextprotocol/server-slack",
        args=["-y", "@modelcontextprotocol/server-slack"],
        description="Send and read Slack messages, manage channels, post notifications.",
        tool_types=["messaging", "communication", "slack", "notifications", "team chat", "alerts"],
        requires_key=True,
        key_name="SLACK_BOT_TOKEN",
        key_url="https://api.slack.com/apps",
        key_instructions="Create a Slack App at api.slack.com/apps, add Bot Token Scopes (chat:write, channels:read), install to workspace.",
    ),
    MCPServerSpec(
        id="google_maps",
        name="Google Maps",
        package="@modelcontextprotocol/server-google-maps",
        args=["-y", "@modelcontextprotocol/server-google-maps"],
        description="Location search, geocoding, directions, and place details via Google Maps.",
        tool_types=["maps", "location", "geocoding", "directions", "address lookup", "places", "geospatial"],
        requires_key=True,
        key_name="GOOGLE_MAPS_API_KEY",
        key_url="https://console.cloud.google.com/",
        key_instructions="Enable Maps JavaScript API + Places API in Google Cloud Console. Free tier: $200/month credit.",
    ),
    MCPServerSpec(
        id="aws_kb",
        name="AWS Knowledge Base",
        package="@modelcontextprotocol/server-aws-kb-retrieval",
        args=["-y", "@modelcontextprotocol/server-aws-kb-retrieval"],
        description="Retrieve documents from AWS Bedrock Knowledge Bases using semantic search.",
        tool_types=["knowledge retrieval", "document search", "RAG", "semantic search", "AWS", "enterprise knowledge"],
        requires_key=True,
        key_name="AWS_ACCESS_KEY_ID",
        key_instructions="Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, and KNOWLEDGE_BASE_ID in your environment.",
    ),
    MCPServerSpec(
        id="everart",
        name="EverArt (Image Generation)",
        package="@modelcontextprotocol/server-everart",
        args=["-y", "@modelcontextprotocol/server-everart"],
        description="Generate images using EverArt AI models.",
        tool_types=["image generation", "image creation", "visual content", "AI art"],
        requires_key=True,
        key_name="EVERART_API_KEY",
        key_url="https://everart.ai",
        key_instructions="Get your API key from everart.ai dashboard.",
    ),
    MCPServerSpec(
        id="sequential_thinking",
        name="Sequential Thinking",
        package="@modelcontextprotocol/server-sequential-thinking",
        args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
        description="Structured multi-step reasoning and problem decomposition tool for complex tasks.",
        tool_types=["reasoning", "problem solving", "planning", "step-by-step thinking", "analysis", "decision making"],
        requires_key=False,
    ),
]

# Fast lookup by tool_type string
def find_servers_for_tool_types(required_tool_types: List[str]) -> List[MCPServerSpec]:
    """Return deduplicated list of MCPServerSpec that cover the given tool types."""
    matched: dict[str, MCPServerSpec] = {}
    for req in required_tool_types:
        req_lower = req.lower().strip()
        req_words = set(req_lower.split())
        for spec in CATALOG:
            if spec.id in matched:
                continue
            # Exact match
            if req_lower in [t.lower() for t in spec.tool_types]:
                matched[spec.id] = spec
                continue
            # Fuzzy word overlap
            for tt in spec.tool_types:
                tt_words = set(tt.lower().split())
                if len(req_words & tt_words) >= 1:
                    matched[spec.id] = spec
                    break
    return list(matched.values())
