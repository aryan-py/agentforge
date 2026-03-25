"""Parses raw web search results into a structured ResearchResult using an LLM."""

import json
import logging
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentforge.schemas.job import JobDefinition
from agentforge.schemas.research import ResearchResult, SearchResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert AI agent architect. Your job is to deeply analyze web research
about a specific role/job and extract EVERYTHING an AI agent would need to perform it well.

Be EXHAUSTIVE and SPECIFIC. Vague answers are useless. Think like a senior engineer building
a production AI agent from scratch.

Extract the following:

1. required_skills — Specific, concrete skills (NOT generic). Examples:
   - BAD: "data analysis"
   - GOOD: "pandas DataFrame manipulation", "SQL window functions", "time-series anomaly detection"
   Aim for 8-15 skills minimum.

2. required_tool_types — Specific tool categories the agent needs. Choose from and extend:
   "web search", "web scraper", "code executor", "database query", "file reader", "file writer",
   "PDF reader", "spreadsheet reader", "calculator", "chart generator", "email sender",
   "Slack notifier", "GitHub API", "REST API caller", "SQL database", "CSV processor",
   "JSON processor", "image analyzer", "calendar access", "task manager", "vector search"
   Be specific — list every tool type the job needs, even if it seems minor.

3. expert_approach — 6-10 concrete steps an expert would follow. Each step must be actionable
   and specific to THIS job, not generic advice.

4. domain_knowledge_summary — A 200-word paragraph of critical domain knowledge the agent must
   have. Include: key concepts, important caveats, common pitfalls, domain-specific terminology.

5. suggested_mcp_servers — List specific MCP server packages, npm packages, or Python libraries
   that directly help with this job. Include:
   - npm MCP packages (e.g. "@modelcontextprotocol/server-github", "@modelcontextprotocol/server-filesystem")
   - Python libraries (e.g. "pandas", "sqlalchemy", "requests")
   - Any specific tools mentioned in the research
   Aim for 5-10 specific packages.

6. confidence — Float 0.0 to 1.0:
   1.0 = found detailed, authoritative step-by-step guides with specific tools
   0.7 = found good general information with some tool mentions
   0.4 = found partial information, mostly generic
   0.2 = found little relevant content

Respond ONLY with valid JSON:
{
  "required_skills": ["skill1", "skill2", ...],
  "required_tool_types": ["type1", "type2", ...],
  "expert_approach": ["step 1...", "step 2...", ...],
  "domain_knowledge_summary": "...",
  "suggested_mcp_servers": ["package1", "package2", ...],
  "confidence": 0.0
}
No text outside the JSON."""


class ResearchResultParser:
    """Extracts structured requirements from raw web search results using an LLM."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    async def parse(
        self, search_results: List[SearchResult], job: JobDefinition
    ) -> ResearchResult:
        """Extract a ResearchResult from web search findings."""

        context_parts = []
        for r in search_results:
            part = f"=== {r.title} ({r.url}) ===\n{r.snippet}"
            if r.full_content:
                part += f"\n{r.full_content[:1500]}"
            context_parts.append(part)
        context = "\n\n".join(context_parts)

        # Use up to 14000 chars of research context
        if len(context) > 14000:
            context = context[:14000]

        user_prompt = f"""Job to analyze:
Type: {job.job_type}
Domain: {job.domain}
Description: {job.description}

=== WEB RESEARCH FINDINGS ({len(search_results)} sources) ===
{context}

Based on all this research, extract a COMPLETE and SPECIFIC profile of what an AI agent
needs to do this job well. Be exhaustive — list every skill, every tool type, every
relevant package you can identify from the research."""

        try:
            response = await self.llm.ainvoke(
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
            )
            raw_json = response.content
            if "```" in raw_json:
                raw_json = raw_json.split("```")[1]
                if raw_json.startswith("json"):
                    raw_json = raw_json[4:]
            data = json.loads(raw_json.strip())
        except Exception as e:
            logger.warning(f"LLM parsing failed: {e}. Using fallback.")
            data = {
                "required_skills": [],
                "required_tool_types": ["web search"],
                "expert_approach": ["Research the topic", "Analyze findings", "Produce output"],
                "domain_knowledge_summary": f"Domain: {job.domain}. Job type: {job.job_type}.",
                "suggested_mcp_servers": [],
                "confidence": 0.1,
            }

        result = ResearchResult(
            job_type=job.job_type,
            domain=job.domain,
            required_skills=data.get("required_skills", []),
            required_tool_types=data.get("required_tool_types", ["web search"]),
            expert_approach=data.get("expert_approach", ["Complete the task systematically"]),
            domain_knowledge_summary=data.get("domain_knowledge_summary", ""),
            suggested_mcp_servers=data.get("suggested_mcp_servers", []),
            search_queries_used=[r.url for r in search_results if r.url],
            sources=[r.url for r in search_results if r.url],
            confidence=float(data.get("confidence", 0.0)),
        )

        if not result.required_skills:
            result.confidence = min(result.confidence, 0.1)
        if not result.required_tool_types:
            result.required_tool_types = ["web search"]
        if not result.expert_approach:
            result.expert_approach = ["Research the topic", "Analyze findings", "Produce output"]

        return result

    async def refine(self, result: ResearchResult, additional_query: str) -> ResearchResult:
        """Run one more targeted search to fill gaps and merge findings."""
        from agentforge.research.web_searcher import WebSearcher

        searcher = WebSearcher(max_results_per_query=5, fetch_pages=True)

        # Run two targeted refinement queries
        extra_results = []
        for q in [additional_query, f"MCP server tools {result.domain} {result.job_type} npm packages"]:
            extra_results.extend(await searcher.search(q))

        from agentforge.schemas.job import JobDefinition

        fake_job = JobDefinition(
            raw_input=additional_query,
            job_type=result.job_type,
            title=result.job_type,
            description=additional_query,
            domain=result.domain,
            output_type="text",
            expected_output="",
            tenant_id="system",
        )
        refined = await self.parse(extra_results, fake_job)

        result.required_skills = list(dict.fromkeys(result.required_skills + refined.required_skills))
        result.required_tool_types = list(
            dict.fromkeys(result.required_tool_types + refined.required_tool_types)
        )
        result.suggested_mcp_servers = list(
            dict.fromkeys(result.suggested_mcp_servers + refined.suggested_mcp_servers)
        )
        result.expert_approach = result.expert_approach or refined.expert_approach
        if not result.domain_knowledge_summary:
            result.domain_knowledge_summary = refined.domain_knowledge_summary
        result.confidence = max(result.confidence, refined.confidence)
        result.sources = list(dict.fromkeys(result.sources + refined.sources))

        return result
