"""AgentForge tools package — registry factory and built-in tool registration."""

import logging

from agentforge.tools.builtin.calculator import TOOL_TYPES as CALC_TYPES
from agentforge.tools.builtin.calculator import calculator
from agentforge.tools.builtin.database import TOOL_TYPES as DB_TYPES
from agentforge.tools.builtin.database import database_query
from agentforge.tools.builtin.file_ops import FILE_READER_TOOL_TYPES, FILE_WRITER_TOOL_TYPES
from agentforge.tools.builtin.file_ops import file_reader, file_writer
from agentforge.tools.builtin.page_reader import TOOL_TYPES as PAGE_TYPES
from agentforge.tools.builtin.page_reader import page_reader
from agentforge.tools.builtin.web_search import TOOL_TYPES as WEB_TYPES
from agentforge.tools.builtin.web_search import web_search
from agentforge.tools.registry import ToolRegistry, ToolRegistryEntry

logger = logging.getLogger(__name__)


def create_default_registry() -> ToolRegistry:
    """Create a ToolRegistry pre-populated with all 6 built-in tools.

    The tool_types lists for each entry must align with what ResearchResultParser
    extracts from web content — this is the critical mapping bridge.
    """
    registry = ToolRegistry()

    entries = [
        ToolRegistryEntry(
            tool_id="web_search",
            name="Web Search",
            description=(
                "Searches the web using DuckDuckGo. Use when you need current information, "
                "facts, news, or general knowledge that requires internet access."
            ),
            tool_types=WEB_TYPES,
            capability_tags=["internet", "duckduckgo", "search engine"],
            source="builtin",
            cost_tier="free",
            langchain_tool=web_search,
        ),
        ToolRegistryEntry(
            tool_id="page_reader",
            name="Page Reader",
            description=(
                "Fetches and reads the full text content of a URL. Use when you have a "
                "specific page or article you need to read in detail."
            ),
            tool_types=PAGE_TYPES,
            capability_tags=["http", "html", "scraper"],
            source="builtin",
            cost_tier="free",
            langchain_tool=page_reader,
        ),
        ToolRegistryEntry(
            tool_id="calculator",
            name="Calculator",
            description=(
                "Evaluates mathematical expressions safely. Use for arithmetic, financial "
                "calculations, or any numeric computation."
            ),
            tool_types=CALC_TYPES,
            capability_tags=["math", "numbers", "formula"],
            source="builtin",
            cost_tier="free",
            langchain_tool=calculator,
        ),
        ToolRegistryEntry(
            tool_id="file_reader",
            name="File Reader",
            description=(
                "Reads text and PDF files from the local filesystem. Use when you need "
                "to process documents, reports, or data files."
            ),
            tool_types=FILE_READER_TOOL_TYPES,
            capability_tags=["filesystem", "pdf", "documents"],
            source="builtin",
            cost_tier="free",
            langchain_tool=file_reader,
        ),
        ToolRegistryEntry(
            tool_id="file_writer",
            name="File Writer",
            description=(
                "Writes content to files on the local filesystem. Use to save reports, "
                "outputs, or any generated content."
            ),
            tool_types=FILE_WRITER_TOOL_TYPES,
            capability_tags=["filesystem", "save", "export"],
            source="builtin",
            cost_tier="free",
            langchain_tool=file_writer,
        ),
        ToolRegistryEntry(
            tool_id="database_query",
            name="Database Query",
            description=(
                "Queries a SQLite database with SELECT statements. Use to retrieve "
                "structured data, run aggregations, or look up records."
            ),
            tool_types=DB_TYPES,
            capability_tags=["sqlite", "sql", "database"],
            source="builtin",
            cost_tier="free",
            langchain_tool=database_query,
        ),
    ]

    for entry in entries:
        registry.register(entry)

    all_types = registry.list_all_tool_types()
    logger.info(f"Tool Registry initialized: {len(entries)} tools, {len(all_types)} tool types covered")
    print(f"Tool Registry initialized: {len(entries)} tools, {len(all_types)} tool types covered")

    return registry


__all__ = ["ToolRegistry", "ToolRegistryEntry", "create_default_registry"]
