"""Generates all config files from a ResearchResult + user choices."""

import json
from typing import List, Literal

import yaml

from agentforge.config_generator.mcp_catalog import MCPServerSpec
from agentforge.schemas.research import ResearchResult

Framework = Literal[
    "claude_desktop",
    "cursor",
    "windsurf",
    "langgraph",
    "langchain",
    "crewai",
    "custom_python",
]

FRAMEWORK_LABELS = {
    "claude_desktop": "Claude Desktop",
    "cursor": "Cursor",
    "windsurf": "Windsurf",
    "langgraph": "LangGraph",
    "langchain": "LangChain",
    "crewai": "CrewAI",
    "custom_python": "Custom Python Agent",
}

# Where each framework expects the MCP config file
FRAMEWORK_CONFIG_PATH = {
    "claude_desktop": "claude_desktop_config.json  →  paste into your Claude Desktop config",
    "cursor": ".cursor/mcp.json  →  place at your project root",
    "windsurf": ".windsurf/mcp_config.json  →  place at your project root",
    "langgraph": "mcp_config.py  →  import in your LangGraph agent",
    "langchain": "mcp_config.py  →  import in your LangChain agent",
    "crewai": "mcp_config.py  →  import in your CrewAI crew",
    "custom_python": "mcp_config.py  →  import in your agent",
}


def _server_env_block(spec: MCPServerSpec) -> dict:
    """Build the env dict for a server config block."""
    env = {}
    if spec.requires_key and spec.key_name:
        env[spec.key_name] = f"YOUR_{spec.key_name}_HERE"
    return env


def generate_mcp_config(servers: List[MCPServerSpec], framework: Framework) -> str:
    """Generate the framework-specific MCP config file content."""

    if framework in ("claude_desktop", "cursor", "windsurf"):
        # JSON format — all three use the same schema, different file paths
        mcp_servers = {}
        for spec in servers:
            entry: dict = {
                "command": spec.command,
                "args": spec.args,
            }
            env = _server_env_block(spec)
            if env:
                entry["env"] = env
            mcp_servers[spec.id.replace("_", "-")] = entry

        config = {"mcpServers": mcp_servers}

        lines = [
            f"// AgentForge generated MCP config — {FRAMEWORK_LABELS[framework]}",
            f"// File location: {FRAMEWORK_CONFIG_PATH[framework]}",
            "//",
        ]

        # Annotate each server
        annotated_lines = []
        raw = json.dumps(config, indent=2).splitlines()

        # Rebuild with inline comments (JSON doesn't support comments natively,
        # so we prefix the file with a comment block listing all servers)
        free_servers = [s for s in servers if not s.requires_key]
        key_servers = [s for s in servers if s.requires_key]

        if free_servers:
            lines.append("// ✅ FREE — no API key required:")
            for s in free_servers:
                lines.append(f"//   • {s.name} ({s.package})")
                if s.extra_placeholders:
                    for note in s.extra_placeholders.values():
                        lines.append(f"//     ⚠️  {note}")
        if key_servers:
            lines.append("//")
            lines.append("// 🔑 REQUIRES API KEY — get yours at the links below:")
            for s in key_servers:
                lines.append(f"//   • {s.name}: {s.key_name}")
                if s.key_url:
                    lines.append(f"//     Get key: {s.key_url}")
                if s.key_instructions:
                    lines.append(f"//     Instructions: {s.key_instructions}")

        lines.append("//")
        lines += raw
        return "\n".join(lines)

    else:
        # Python dict format for LangGraph / LangChain / CrewAI / custom
        lines = [
            '"""',
            f"AgentForge generated MCP config — {FRAMEWORK_LABELS[framework]}",
            f"File location: {FRAMEWORK_CONFIG_PATH[framework]}",
            '"""',
            "",
        ]

        free_servers = [s for s in servers if not s.requires_key]
        key_servers = [s for s in servers if s.requires_key]

        if free_servers:
            lines.append("# ✅ FREE — no API key required:")
            for s in free_servers:
                lines.append(f"#   • {s.name} ({s.package})")
                if s.extra_placeholders:
                    for note in s.extra_placeholders.values():
                        lines.append(f"#   ⚠️  {note}")
        if key_servers:
            lines.append("#")
            lines.append("# 🔑 REQUIRES API KEY — replace the placeholder values below:")
            for s in key_servers:
                lines.append(f"#   • {s.name}: set {s.key_name}")
                if s.key_url:
                    lines.append(f"#     Get key: {s.key_url}")
                if s.key_instructions:
                    lines.append(f"#     {s.key_instructions}")

        lines.append("")
        lines.append("import os")
        lines.append("")
        lines.append("MCP_SERVERS = [")

        for spec in servers:
            lines.append(f"    # {'✅ FREE' if not spec.requires_key else '🔑 NEEDS KEY'} — {spec.name}")
            lines.append(f"    # {spec.description}")
            entry = {
                "name": spec.id.replace("_", "-"),
                "transport": "stdio",
                "command": spec.command,
                "args": spec.args,
            }
            if spec.requires_key and spec.key_name:
                entry["env"] = {spec.key_name: f"os.environ.get('{spec.key_name}', 'YOUR_{spec.key_name}_HERE')"}
            lines.append(f"    {json.dumps(entry, indent=4)},".replace('"', '"'))
            lines.append("")

        lines.append("]")
        lines.append("")

        if framework in ("langgraph", "langchain"):
            lines += [
                "",
                "# ── Usage with langchain-mcp-adapters ──────────────────────────",
                "# from langchain_mcp_adapters.client import MultiServerMCPClient",
                "#",
                "# async with MultiServerMCPClient({",
                '#     s["name"]: {k: v for k, v in s.items() if k != "name"}',
                "#     for s in MCP_SERVERS",
                "# }) as client:",
                "#     tools = client.get_tools()",
                "#     # pass tools to your agent",
            ]
        elif framework == "crewai":
            lines += [
                "",
                "# ── Usage with CrewAI ────────────────────────────────────────────",
                "# from crewai_tools import MCPServerAdapter",
                "#",
                "# tools = []",
                "# for server_config in MCP_SERVERS:",
                "#     adapter = MCPServerAdapter(server_config)",
                "#     tools.extend(adapter.tools)",
            ]
        else:
            lines += [
                "",
                "# ── Usage (generic) ─────────────────────────────────────────────",
                "# Pass MCP_SERVERS to your MCP client of choice.",
                "# Each entry has: name, transport, command, args, (optional) env.",
            ]

        return "\n".join(lines)


def generate_system_prompt(research: ResearchResult, role: str = "") -> str:
    """Generate a ready-to-use system prompt pre-filled with research findings."""
    skills_list = "\n".join(f"- {s}" for s in research.required_skills)
    approach_list = "\n".join(f"{i+1}. {step}" for i, step in enumerate(research.expert_approach))
    role_title = role or f"{research.domain.title()} Specialist"

    return f"""# System Prompt — {role_title}
# Generated by AgentForge from web research
# ─────────────────────────────────────────────────────────────────
# HOW TO USE:
#   Paste everything below the dashed line into your agent's system prompt.
#   Review and adjust the ROLE, DOMAIN KNOWLEDGE, and CONSTRAINTS sections
#   to fit your specific use case.
# ─────────────────────────────────────────────────────────────────

<role>
You are a {role_title} specialising in {research.domain}.
</role>

<domain_knowledge>
{research.domain_knowledge_summary}
</domain_knowledge>

<skills>
You have deep expertise in the following areas:
{skills_list}
</skills>

<approach>
When completing tasks in this domain, follow this expert approach:
{approach_list}
</approach>

<constraints>
- Always verify information using your tools before stating it as fact.
- If a task falls outside your capabilities or tool set, say so clearly.
- Prefer accuracy over speed — double-check calculations and data.
- [ADD YOUR OWN CONSTRAINTS HERE]
</constraints>
"""


def generate_skills_yaml(research: ResearchResult) -> str:
    """Generate a structured skills.yaml describing the agent's capability profile."""
    data = {
        "agent_profile": {
            "domain": research.domain,
            "job_type": research.job_type,
            "research_confidence": round(research.confidence, 2),
            "generated_by": "AgentForge",
        },
        "skills": research.required_skills,
        "tool_types_needed": research.required_tool_types,
        "expert_approach": research.expert_approach,
        "domain_knowledge_summary": research.domain_knowledge_summary,
        "suggested_mcp_servers": research.suggested_mcp_servers,
    }
    return "# AgentForge generated skills profile\n# Review and adjust before committing.\n\n" + yaml.dump(
        data, default_flow_style=False, allow_unicode=True, sort_keys=False
    )


def generate_requirements_txt(servers: List[MCPServerSpec], research: ResearchResult) -> str:
    """Generate a requirements.txt with Python packages for the agent stack."""
    packages = set()

    # Core packages always needed
    packages.update([
        "langchain>=0.3",
        "langchain-core",
        "langgraph>=0.2",
        "langchain-mcp-adapters",
        "pydantic>=2.0",
        "httpx",
    ])

    # Domain-specific packages inferred from skills
    skill_text = " ".join(research.required_skills + research.required_tool_types).lower()
    if any(w in skill_text for w in ["anthropic", "claude"]):
        packages.add("langchain-anthropic")
    if any(w in skill_text for w in ["openai", "gpt"]):
        packages.add("langchain-openai")
    if any(w in skill_text for w in ["pdf", "document"]):
        packages.add("pypdf")
    if any(w in skill_text for w in ["sql", "database", "sqlite"]):
        packages.add("sqlalchemy")
    if any(w in skill_text for w in ["data", "analysis", "csv", "spreadsheet"]):
        packages.add("pandas")
    if any(w in skill_text for w in ["chart", "visualization", "plot"]):
        packages.add("plotly")
    if any(w in skill_text for w in ["web scraping", "html", "beautifulsoup"]):
        packages.add("beautifulsoup4")
        packages.add("lxml")

    # Packages from MCP server specs
    for spec in servers:
        packages.update(spec.python_packages)

    # Default LLM provider if nothing detected
    if not any(p.startswith("langchain-anthropic") or p.startswith("langchain-openai") for p in packages):
        packages.add("langchain-anthropic")

    sorted_pkgs = sorted(packages)
    lines = [
        "# AgentForge generated requirements",
        "# Install with: pip install -r requirements.txt",
        "# Note: MCP servers run via npx — ensure Node.js >= 18 is installed.",
        "",
    ]
    lines += sorted_pkgs
    return "\n".join(lines)


def generate_readme(
    research: ResearchResult,
    servers: List[MCPServerSpec],
    framework: Framework,
) -> str:
    """Generate a README.md with setup instructions for the generated config."""
    free = [s for s in servers if not s.requires_key]
    keyed = [s for s in servers if s.requires_key]

    lines = [
        f"# Agent Setup — {research.domain.title()} / {research.job_type}",
        "",
        f"> Generated by AgentForge for framework: **{FRAMEWORK_LABELS[framework]}**",
        "",
        "## What this agent does",
        "",
        research.domain_knowledge_summary,
        "",
        "## Files in this package",
        "",
        f"| File | Purpose |",
        f"|------|---------|",
        f"| `mcp_config{'_' + framework if framework not in ('claude_desktop','cursor','windsurf') else ''}.json/.py` | MCP server configuration for {FRAMEWORK_LABELS[framework]} |",
        f"| `system_prompt.txt` | Ready-to-use system prompt — paste into your agent |",
        f"| `skills.yaml` | Structured skills and capability profile |",
        f"| `requirements.txt` | Python dependencies |",
        "",
        "## Setup",
        "",
        "### 1. Install Python dependencies",
        "```bash",
        "pip install -r requirements.txt",
        "```",
        "",
        "### 2. Install Node.js (required for MCP servers)",
        "MCP servers run via `npx`. Ensure Node.js >= 18 is installed:",
        "```bash",
        "node --version  # should be >= 18",
        "```",
    ]

    if keyed:
        lines += [
            "",
            "### 3. Set API keys",
            "",
            "The following MCP servers require API keys. Set them as environment variables:",
            "",
        ]
        for s in keyed:
            lines.append(f"#### {s.name}")
            lines.append(f"```bash")
            lines.append(f'export {s.key_name}="your-key-here"')
            lines.append(f"```")
            if s.key_url:
                lines.append(f"Get your key: {s.key_url}")
            if s.key_instructions:
                lines.append(f"> {s.key_instructions}")
            lines.append("")

    if free:
        lines += [
            "",
            f"### {'4' if keyed else '3'}. Free MCP servers (no setup needed)",
            "",
        ]
        for s in free:
            lines.append(f"- **{s.name}** (`{s.package}`) — {s.description}")
            if s.extra_placeholders:
                for note in s.extra_placeholders.values():
                    lines.append(f"  - ⚠️  {note}")

    lines += [
        "",
        "## Skills this agent has",
        "",
    ]
    for skill in research.required_skills:
        lines.append(f"- {skill}")

    lines += [
        "",
        "## Expert approach",
        "",
    ]
    for i, step in enumerate(research.expert_approach, 1):
        lines.append(f"{i}. {step}")

    lines += [
        "",
        "---",
        "*Generated by [AgentForge](https://github.com/agentforge)*",
    ]

    return "\n".join(lines)
