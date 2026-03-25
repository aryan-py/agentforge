"""Web search and page fetching for the Trainer's research phase."""

import asyncio
import logging
from typing import List

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from agentforge.schemas.research import SearchResult

logger = logging.getLogger(__name__)

_NOISE_CLASSES = {"nav", "footer", "cookie", "ad", "sidebar", "menu", "header", "banner", "popup"}


class WebSearcher:
    """Searches the web and fetches page content for the TrainerAgent.

    Uses DuckDuckGo (no API key required) to run multiple focused queries
    about a job's requirements, then optionally fetches and cleans the top
    pages for richer content.
    """

    def __init__(self, max_results_per_query: int = 6, fetch_pages: bool = True):
        self.max_results_per_query = max_results_per_query
        self.fetch_pages = fetch_pages

    async def search(self, query: str) -> List[SearchResult]:
        """Run a single DuckDuckGo search and optionally fetch top pages."""
        logger.info(f"🔍 Searching: {query}")
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=self.max_results_per_query))
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")
            return []

        results: List[SearchResult] = []
        for item in raw:
            results.append(
                SearchResult(
                    url=item.get("href", ""),
                    title=item.get("title", ""),
                    snippet=item.get("body", ""),
                )
            )

        if self.fetch_pages:
            # Fetch full content for top 4 results
            fetch_tasks = [self.fetch_page(r.url) for r in results[:4] if r.url]
            contents = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for result, content in zip(results[:4], contents):
                if isinstance(content, str):
                    result.full_content = content

        return results

    async def fetch_page(self, url: str) -> str:
        """Fetch and clean the text content of a web page.

        Strips navigation, footers, ads, and other noise. Returns at most
        5000 characters. Returns empty string on any error.
        """
        try:
            async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
                response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()
                html = response.text
        except Exception as e:
            logger.debug(f"Failed to fetch {url}: {e}")
            return ""

        try:
            soup = BeautifulSoup(html, "lxml")

            for tag in soup.find_all(True):
                classes = " ".join(tag.get("class", [])).lower()
                tag_name = tag.name.lower() if tag.name else ""
                if any(noise in classes for noise in _NOISE_CLASSES) or tag_name in (
                    "nav", "footer", "header", "aside", "script", "style", "noscript",
                ):
                    tag.decompose()

            parts = []
            for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "code", "pre"]):
                text = tag.get_text(separator=" ", strip=True)
                if text and len(text) > 20:
                    parts.append(text)

            cleaned = "\n".join(parts)
            return cleaned[:5000]
        except Exception as e:
            logger.debug(f"Failed to parse {url}: {e}")
            return ""

    async def research_job(
        self, job_type: str, domain: str, job_description: str
    ) -> List[SearchResult]:
        """Run multiple focused queries to deeply research what a job requires.

        Runs 10 queries across skills, tools, MCP servers, workflows, and
        best practices. Returns all results deduplicated by URL.
        """
        queries = [
            # Core skills
            f"what skills are required for {job_type} in {domain}",
            f"{domain} {job_type} required knowledge and expertise 2024 2025",
            # Tools & libraries
            f"best tools and software for {job_type} {domain} professionals",
            f"Python libraries and packages for {domain} {job_type} automation",
            # MCP servers & AI tools
            f"MCP server tools for {domain} {job_type} AI agent",
            f"npm MCP packages {domain} automation tools",
            # Workflows & best practices
            f"how to automate {job_description[:80]} step by step guide",
            f"{domain} expert workflow {job_type} best practices tutorial",
            # AI agent specific
            f"AI agent {domain} {job_type} tools plugins integrations",
            # APIs & integrations
            f"{domain} APIs and integrations for {job_type} 2024 2025",
        ]

        all_results: List[SearchResult] = []
        seen_urls: set[str] = set()

        for query in queries:
            results = await self.search(query)
            for r in results:
                if r.url and r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)
            await asyncio.sleep(0.4)

        logger.info(
            f"📄 Research complete: {len(all_results)} unique sources from {len(queries)} queries"
        )
        return all_results
