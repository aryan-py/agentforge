"""Built-in web search tool using DuckDuckGo."""

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for information using DuckDuckGo.

    Use this tool to find current information, facts, news, and general knowledge.
    Input should be a clear search query string.
    Returns the top 5 search results as formatted text.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"No results found for: {query}"

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"{i}. **{r.get('title', 'No title')}**\n"
                f"   URL: {r.get('href', '')}\n"
                f"   {r.get('body', '')}"
            )
        return "\n\n".join(parts)
    except Exception as e:
        return f"Search failed: {e}"


# Registry metadata
TOOL_TYPES = [
    "web search",
    "real-time information",
    "fact lookup",
    "current events",
    "research",
    "news",
    "internet search",
    "online search",
]
