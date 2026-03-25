"""TrainerAgent — the central piece that researches what a job needs and configures the Worker."""

import logging
import time
from typing import List, Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agentforge.config.settings import settings
from agentforge.research.cache import ResearchCache
from agentforge.research.mcp_discoverer import MCPDiscoverer, MCPServerSuggestion
from agentforge.research.result_parser import ResearchResultParser
from agentforge.research.web_searcher import WebSearcher
from agentforge.schemas.context_pack import ContextPack
from agentforge.schemas.job import JobDefinition
from agentforge.schemas.research import ResearchResult
from agentforge.tools.registry import MatchResult, ToolRegistry, ToolRegistryEntry

logger = logging.getLogger(__name__)


class TrainingSession(BaseModel):
    """The complete output of a TrainerAgent.train() call.

    Contains everything produced during the training phase: the research findings,
    the worker configuration, tool matching details, and a human-readable log of
    every decision made.
    """

    job_id: str
    research: ResearchResult
    context_pack: ContextPack
    match_result: MatchResult
    training_log: List[str]
    duration_seconds: float

    model_config = {"arbitrary_types_allowed": True}


class TrainerAgent:
    """Researches what a job requires and builds a complete Worker Agent configuration.

    The Trainer never executes jobs. Its sole purpose is to:
    1. Research what the job requires (web search or cache/episodic memory)
    2. Map research findings to registered tools
    3. Build the Worker's ContextPack from research data

    Every decision is recorded in TrainingSession.training_log for full transparency.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        web_searcher: WebSearcher,
        result_parser: ResearchResultParser,
        research_cache: ResearchCache,
        mcp_discoverer: MCPDiscoverer,
        llm_planner: BaseChatModel,
        llm_fast: BaseChatModel,
        episodic_memory=None,
    ):
        self.tool_registry = tool_registry
        self.web_searcher = web_searcher
        self.result_parser = result_parser
        self.research_cache = research_cache
        self.mcp_discoverer = mcp_discoverer
        self.llm_planner = llm_planner
        self.llm_fast = llm_fast
        self.episodic_memory = episodic_memory

    async def train(self, job: JobDefinition) -> TrainingSession:
        """Research the job, match tools, and build a ContextPack for the Worker."""
        start = time.time()
        log: List[str] = []

        # ── STAGE 1: RESEARCH ────────────────────────────────────────────────
        research = await self._stage_research(job, log)

        # ── STAGE 2: TOOL MATCHING ───────────────────────────────────────────
        match, final_tools = await self._stage_tool_matching(job, research, log)

        # ── STAGE 3: WORKER CONFIGURATION ───────────────────────────────────
        context_pack = await self._stage_worker_config(job, research, final_tools, log)

        duration = time.time() - start
        log.append(f"✅ Training complete in {duration:.1f}s. Worker ready for: '{job.title}'")

        return TrainingSession(
            job_id=job.job_id,
            research=research,
            context_pack=context_pack,
            match_result=match,
            training_log=log,
            duration_seconds=duration,
        )

    # ── Private stage helpers ─────────────────────────────────────────────────

    async def _stage_research(self, job: JobDefinition, log: List[str]) -> ResearchResult:
        # Check episodic memory first
        if self.episodic_memory:
            past = await self.episodic_memory.recall(job)
            if past:
                best = max(past, key=lambda x: x.quality_score)
                log.append(f"🧠 Found {len(past)} similar past jobs")
                log.append(f"   Best match: '{best.job_type}/{best.domain}' (quality: {best.quality_score:.0%})")
                if best.lessons:
                    log.append(f"   Lessons: {'; '.join(best.lessons[:2])}")

                if best.quality_score >= 0.8:
                    log.append("⚡ High-quality experience found — skipping web research")
                    from agentforge.schemas.research import ResearchResult
                    research = ResearchResult(
                        job_type=job.job_type,
                        domain=job.domain,
                        required_skills=best.skills_discovered,
                        required_tool_types=best.tool_types_needed,
                        expert_approach=best.lessons + [best.approach_summary],
                        domain_knowledge_summary=f"Based on past experience: {best.approach_summary}",
                        confidence=best.quality_score,
                    )
                    return research

        cached = await self.research_cache.get(job.job_type, job.domain)
        if cached and cached.confidence >= 0.7:
            log.append(
                f"📦 Using cached research for {job.job_type}/{job.domain} "
                f"(confidence: {cached.confidence:.0%})"
            )
            return cached

        log.append(f"🌐 No cache hit — starting web research for '{job.title}'")
        log.append(f"🔍 Researching: what does '{job.job_type}' in '{job.domain}' require?")

        search_results = await self.web_searcher.research_job(
            job.job_type, job.domain, job.description
        )
        log.append(f"📄 Found {len(search_results)} sources")

        research = await self.result_parser.parse(search_results, job)
        log.append(f"🧠 Research complete. Confidence: {research.confidence:.0%}")
        log.append(f"   Skills found: {', '.join(research.required_skills[:5])}")
        log.append(f"   Tool types needed: {', '.join(research.required_tool_types)}")

        if research.confidence < 0.4:
            log.append("⚠️  Low confidence. Running refinement search...")
            refined_query = f"{job.domain} {job.job_type} detailed guide"
            research = await self.result_parser.refine(research, refined_query)
            log.append(f"   Refined confidence: {research.confidence:.0%}")

        await self.research_cache.set(job.job_type, job.domain, research)
        log.append("💾 Research cached for future jobs")
        return research

    async def _stage_tool_matching(
        self, job: JobDefinition, research: ResearchResult, log: List[str]
    ) -> tuple[MatchResult, List[ToolRegistryEntry]]:
        match = self.tool_registry.find_for_requirements(
            research.required_tool_types, job.tenant_id
        )
        log.append(
            f"🔧 Tool matching: {len(match.matched)} tools found, {len(match.unmatched)} gaps"
        )
        for tool in match.matched:
            log.append(f"   ✅ '{tool.name}' covers: {tool.tool_types[:3]}")
        for gap in match.unmatched:
            log.append(f"   ❌ No tool for: '{gap}'")

        # Handle gaps with MCP discovery
        if match.unmatched and settings.ENABLE_MCP_DISCOVERY:
            log.append("🔭 Searching for MCP servers to fill gaps...")
            all_suggestions: List[MCPServerSuggestion] = []
            for gap in match.unmatched:
                suggestions = await self.mcp_discoverer.find_servers_for_capability(gap)
                if suggestions:
                    log.append(f"   💡 For '{gap}': try '{suggestions[0].package_name}'")
                    log.append(f"      Install: {suggestions[0].install_command}")
                    all_suggestions.extend(suggestions)
            if all_suggestions:
                await self.mcp_discoverer.save_suggestions(all_suggestions, job.job_id)
                log.append("📝 MCP suggestions saved to logs/mcp_suggestions.jsonl")

        # Always include web_search as a fallback
        final_tools = list(match.matched)
        tool_ids = {t.tool_id for t in final_tools}
        if "web_search" not in tool_ids:
            fallback = self.tool_registry._entries.get("web_search")
            if fallback:
                final_tools.append(fallback)
                log.append("   ➕ Added web_search as universal fallback")

        log.append(f"🎯 Final tools for worker: {[t.tool_id for t in final_tools]}")
        return match, final_tools

    async def _stage_worker_config(
        self,
        job: JobDefinition,
        research: ResearchResult,
        final_tools: List[ToolRegistryEntry],
        log: List[str],
    ) -> ContextPack:
        system_prompt = self._build_system_prompt(job, research, final_tools)
        estimated_tokens = len(system_prompt) // 4
        log.append(f"📝 System prompt built (~{estimated_tokens} tokens estimated)")

        model = self._select_model(job, research)
        if model == settings.PLANNER_MODEL:
            log.append("🤖 Using Opus (critical priority or low-confidence research)")
        else:
            log.append("🤖 Using Sonnet")

        risk = self._assess_risk(job, research)
        log.append(f"⚠️  Risk level: {risk}")

        role = await self._extract_role(research)
        criteria = await self._derive_criteria(job, research)

        return ContextPack(
            system_prompt=system_prompt,
            role=role,
            goal=job.description,
            expert_approach=research.expert_approach,
            tools=[t.tool_id for t in final_tools],
            model=model,
            knowledge_context=research.domain_knowledge_summary,
            success_criteria=criteria,
            risk_level=risk,
            research_confidence=research.confidence,
        )

    # ── Helper methods ────────────────────────────────────────────────────────

    def _build_system_prompt(
        self,
        job: JobDefinition,
        research: ResearchResult,
        tools: List[ToolRegistryEntry],
    ) -> str:
        skills_list = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(research.required_skills)
        )
        approach_list = "\n".join(
            f"{i+1}. {step}" for i, step in enumerate(research.expert_approach)
        )
        tools_table = "\n".join(
            f"- {t.name} | covers: {', '.join(t.tool_types[:3])} | {t.description[:80]}"
            for t in tools
        )
        constraints_list = "\n".join(f"- {c}" for c in job.constraints)
        if constraints_list:
            constraints_list += "\n"
        constraints_list += (
            "- Always use your tools to verify information before stating it as fact\n"
            "- If you encounter a task your tools cannot handle, say so clearly"
        )

        return f"""<role>
You are a specialist in {job.domain}.
</role>

<goal>
{job.description}
</goal>

<domain_knowledge>
{research.domain_knowledge_summary}
</domain_knowledge>

<required_skills>
You have been equipped with these skills relevant to this job:
{skills_list}
</required_skills>

<expert_approach>
An expert would complete this job by following these steps:
{approach_list}
Follow this approach unless you discover a better path.
</expert_approach>

<available_tools>
{tools_table}
</available_tools>

<constraints>
{constraints_list}
</constraints>"""

    def _select_model(self, job: JobDefinition, research: ResearchResult) -> str:
        if job.priority == "critical" or research.confidence < 0.5:
            return settings.PLANNER_MODEL
        return settings.DEFAULT_MODEL

    def _assess_risk(
        self, job: JobDefinition, research: ResearchResult
    ) -> Literal["low", "medium", "high"]:
        desc_lower = job.description.lower()
        if (
            job.domain.lower() == "legal"
            or any(w in desc_lower for w in ["delete", "send", "payment", "transfer"])
            or (job.domain.lower() == "finance" and job.output_type == "action")
        ):
            return "high"
        if job.domain.lower() == "hr" or research.confidence < 0.4:
            return "medium"
        return "low"

    async def _extract_role(self, research: ResearchResult) -> str:
        if not research.required_skills:
            return "General AI Assistant"
        skills_str = ", ".join(research.required_skills[:5])
        try:
            response = await self.llm_fast.ainvoke(
                [
                    SystemMessage(
                        content="Generate a single professional role title (5-8 words) "
                        "for an expert with these skills. Return ONLY the title, nothing else."
                    ),
                    HumanMessage(content=f"Skills: {skills_str}"),
                ]
            )
            return response.content.strip().strip('"').strip("'")
        except Exception:
            return f"{research.domain.title()} Specialist"

    async def _derive_criteria(
        self, job: JobDefinition, research: ResearchResult
    ) -> List[str]:
        try:
            response = await self.llm_fast.ainvoke(
                [
                    SystemMessage(
                        content="Generate exactly 3 measurable success criteria for this job. "
                        "Return as a JSON array of strings only. No markdown, no explanation."
                    ),
                    HumanMessage(
                        content=f"Job: {job.description}\nDomain: {job.domain}\n"
                        f"Skills needed: {', '.join(research.required_skills[:5])}"
                    ),
                ]
            )
            import json
            raw = response.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception:
            return [
                f"The output addresses all aspects of: {job.description[:60]}",
                "The output is accurate and based on verified information",
                "The output is well-structured and ready to use",
            ]
