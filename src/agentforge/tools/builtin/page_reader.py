"""Built-in page reader tool — fetches and cleans a URL's content."""

from langchain_core.tools import tool


@tool
def page_reader(url: str) -> str:
    """Fetch and read the text content of a web page.

    Use this tool when you have a specific URL and need to read its full content.
    Input should be a valid URL string.
    Returns cleaned text content (up to 4000 characters).
    """
    try:
        import httpx
        from bs4 import BeautifulSoup

        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        for tag in soup.find_all(["nav", "footer", "header", "aside", "script", "style"]):
            tag.decompose()

        parts = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            text = tag.get_text(separator=" ", strip=True)
            if text:
                parts.append(text)

        content = "\n".join(parts)[:4000]
        return content if content else f"No readable content found at {url}"
    except Exception as e:
        return f"Failed to read page: {e}"


TOOL_TYPES = [
    "web page reader",
    "URL reader",
    "article reader",
    "document fetcher",
    "content extraction",
    "page scraper",
    "link reader",
]
