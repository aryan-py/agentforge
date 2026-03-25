"""Packages all generated config files into a single downloadable ZIP."""

import io
import zipfile
from typing import List

from agentforge.config_generator.generator import (
    Framework,
    generate_mcp_config,
    generate_readme,
    generate_requirements_txt,
    generate_skills_yaml,
    generate_system_prompt,
)
from agentforge.config_generator.mcp_catalog import MCPServerSpec
from agentforge.schemas.research import ResearchResult

# Framework → MCP config filename inside the ZIP
_CONFIG_FILENAME = {
    "claude_desktop": "claude_desktop_config.json",
    "cursor": ".cursor/mcp.json",
    "windsurf": ".windsurf/mcp_config.json",
    "langgraph": "mcp_config.py",
    "langchain": "mcp_config.py",
    "crewai": "mcp_config.py",
    "custom_python": "mcp_config.py",
}


def build_zip(
    research: ResearchResult,
    servers: List[MCPServerSpec],
    framework: Framework,
    role: str = "",
) -> bytes:
    """Build a ZIP archive containing all generated config files.

    Returns raw bytes suitable for st.download_button.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # MCP config
        mcp_content = generate_mcp_config(servers, framework)
        zf.writestr(_CONFIG_FILENAME[framework], mcp_content)

        # System prompt
        zf.writestr("system_prompt.txt", generate_system_prompt(research, role))

        # Skills YAML
        zf.writestr("skills.yaml", generate_skills_yaml(research))

        # Requirements
        zf.writestr("requirements.txt", generate_requirements_txt(servers, research))

        # README
        zf.writestr("README.md", generate_readme(research, servers, framework))

    return buf.getvalue()
