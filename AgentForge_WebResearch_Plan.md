# AgentForge — Web-Research-First Build Plan
### The trainer agent searches the web to discover what every job needs, then builds the worker from scratch.

---

## The Core Idea (Read This First)

When a job arrives, **no agent is pre-configured for it**. Instead:

```
Job arrives
    ↓
Trainer Agent searches the web:
  "What skills does a [job type] need?"
  "What tools are used for [job domain]?"
  "What knowledge does an expert in [domain] have?"
    ↓
Trainer reads results, extracts:
  → Required skills list
  → Required tool types
  → Domain knowledge summary
  → Step-by-step approach an expert would take
    ↓
Trainer cross-references against tool registry:
  → Finds matching tools for each requirement
  → Identifies gaps
  → Searches for MCP servers to fill gaps
    ↓
Trainer builds the Worker Agent:
  → Writes a custom system prompt from research
  → Assigns exactly the tools the job needs
  → Injects domain knowledge as context
    ↓
Worker Agent executes the job with its new capabilities
    ↓
Result + lessons stored → next similar job skips research
```

**Nothing is hardcoded.** The trainer discovers everything at runtime. A job about "maritime insurance claims" gets researched the same way as "Python code review" — the trainer figures both out from scratch.

---

## How to Use This Document

Each step has:
- **What you're building** — the goal
- **Claude Code Prompt** — paste directly into `claude` terminal
- **Verify** — commands to confirm it worked before moving on

Start Claude Code from your project root: `claude`

---

# PHASE 1 — FOUNDATION
## The Research Engine & Core Pipeline

---

## Step 1.1 — Project Setup & Schemas

### What you're building
The full folder structure and every Pydantic data model the system uses. Pay attention to `ResearchResult` — this is new and central to everything.

### Claude Code Prompt

```
Create the AgentForge project from scratch. This system's core mechanic is:
a Trainer Agent that searches the web to figure out what any job requires,
then builds a Worker Agent with exactly those capabilities.

1. pyproject.toml with dependencies:
   langchain>=0.3, langchain-anthropic, langchain-openai, langchain-community
   langgraph>=0.2
   langchain-mcp-adapters
   pydantic>=2.0, pydantic-settings, pydantic-ai
   fastapi, uvicorn[standard]
   qdrant-client
   langfuse
   temporalio
   httpx, aiohttp
   duckduckgo-search
   beautifulsoup4, lxml
   pypdf
   tqdm
   pytest, pytest-asyncio, pytest-mock (dev)

2. Full folder structure — create all these as empty files with docstring stubs:
   src/agentforge/__init__.py
   src/agentforge/research/__init__.py
   src/agentforge/research/web_searcher.py
   src/agentforge/research/result_parser.py
   src/agentforge/research/cache.py
   src/agentforge/research/mcp_discoverer.py
   src/agentforge/core/__init__.py
   src/agentforge/core/trainer.py
   src/agentforge/core/worker.py
   src/agentforge/core/pipeline.py
   src/agentforge/core/meta_agent.py
   src/agentforge/core/evaluator.py
   src/agentforge/tools/__init__.py
   src/agentforge/tools/registry.py
   src/agentforge/tools/resolver.py
   src/agentforge/tools/builtin/__init__.py
   src/agentforge/tools/builtin/web_search.py
   src/agentforge/tools/builtin/page_reader.py
   src/agentforge/tools/builtin/calculator.py
   src/agentforge/tools/builtin/file_ops.py
   src/agentforge/tools/builtin/database.py
   src/agentforge/memory/__init__.py
   src/agentforge/memory/episodic.py
   src/agentforge/memory/research_cache.py
   src/agentforge/prompts/__init__.py
   src/agentforge/prompts/builder.py
   src/agentforge/api/__init__.py
   src/agentforge/api/app.py
   src/agentforge/api/routes/jobs.py
   src/agentforge/api/routes/health.py
   src/agentforge/config/__init__.py
   src/agentforge/config/settings.py
   tests/__init__.py
   tests/unit/__init__.py
   tests/integration/__init__.py

3. In src/agentforge/schemas/ create these Pydantic v2 models:

   --- job.py ---
   JobDefinition:
     job_id: str = Field(default_factory=lambda: str(uuid4()))
     raw_input: str                    # the original unprocessed request
     job_type: str                     # inferred by meta-agent
     title: str
     description: str
     domain: str
     output_type: Literal["text","structured","file","action"]
     expected_output: str
     constraints: List[str] = []
     priority: Literal["low","medium","high","critical"] = "medium"
     timeout_seconds: int = 300
     tenant_id: str
     requester_id: str = "api"
     created_at: datetime = Field(default_factory=datetime.utcnow)

   JobResult:
     job_id: str
     status: Literal["pending","researching","training","running","success","failed","review"]
     output: Optional[Any] = None
     quality_score: Optional[float] = None
     error: Optional[str] = None
     tools_used: List[str] = []
     research_queries: List[str] = []   # what the trainer searched for
     skills_discovered: List[str] = []  # what research found
     tokens_used: int = 0
     cost_usd: float = 0.0
     duration_seconds: float = 0.0

   --- research.py ---
   SearchResult:
     url: str
     title: str
     snippet: str
     full_content: Optional[str] = None  # fetched page content

   ResearchResult:
     job_type: str
     domain: str
     required_skills: List[str]
       # e.g. ["financial modeling","data visualization","regulatory knowledge"]
     required_tool_types: List[str]
       # e.g. ["spreadsheet","chart generator","document reader","calculator"]
     expert_approach: List[str]
       # step-by-step: how an expert human would tackle this job
     domain_knowledge_summary: str
       # paragraph of key domain context the worker agent should know
     suggested_mcp_servers: List[str] = []
       # MCP package names that would help: e.g. "@modelcontextprotocol/server-sqlite"
     search_queries_used: List[str] = []
     sources: List[str] = []
     confidence: float = 0.0           # 0-1, how thorough the research was
     researched_at: datetime = Field(default_factory=datetime.utcnow)

   --- context_pack.py ---
   ContextPack:
     # Everything the Worker needs — assembled entirely from research
     system_prompt: str
     role: str
     goal: str
     expert_approach: List[str]        # from research — step by step plan
     tools: List[str]                  # tool_ids to give the worker
     model: str = "claude-sonnet-4-5-20251001"
     temperature: float = 0.1
     max_iterations: int = 15
     knowledge_context: str            # domain knowledge from research
     success_criteria: List[str]
     risk_level: Literal["low","medium","high"] = "low"
     research_confidence: float = 0.0
     estimated_cost_usd: Optional[float] = None

4. In src/agentforge/config/settings.py (Pydantic BaseSettings):
   ANTHROPIC_API_KEY: str
   OPENAI_API_KEY: str
   QDRANT_URL: str = "http://localhost:6333"
   LANGFUSE_PUBLIC_KEY: str = ""
   LANGFUSE_SECRET_KEY: str = ""
   DEFAULT_MODEL: str = "claude-sonnet-4-5-20251001"
   PLANNER_MODEL: str = "claude-opus-4-5-20251101"
   ROUTER_MODEL: str = "gpt-4o-mini"
   RESEARCH_CACHE_TTL_DAYS: int = 7
   MAX_SEARCH_QUERIES_PER_JOB: int = 5
   MAX_PAGES_TO_FETCH: int = 3
   ENABLE_MCP_DISCOVERY: bool = True

5. Create .env.example with all settings and helpful comments explaining each one.

Make every model fully typed. Add a docstring to every class explaining its role
in the research-first pipeline.
```

### Verify
```bash
uv run python -c "from agentforge.schemas.research import ResearchResult; print('OK')"
```

---

## Step 1.2 — The Web Research Engine

### What you're building
The `WebSearcher` and `ResultParser` — the two classes that power the trainer's ability to discover what any job needs. The searcher finds pages, the parser extracts structured requirements from them.

### Claude Code Prompt

```
Build the web research engine in src/agentforge/research/.
This is the core of AgentForge — it's how the trainer figures out what
any job requires by reading the web.

1. In src/agentforge/research/web_searcher.py:

   WebSearcher class:
   
   __init__(max_results_per_query: int = 5, fetch_pages: bool = True):
     Uses duckduckgo_search.DDGS for free searches with no API key required.
     fetch_pages=True means it follows top links and reads full page content.

   async search(query: str) -> List[SearchResult]:
     - Calls DDGS().text(query, max_results=max_results_per_query)
     - For the top 3 results (if fetch_pages=True), calls fetch_page(url)
     - Returns List[SearchResult] with title, url, snippet, full_content

   async fetch_page(url: str) -> str:
     - Uses httpx.AsyncClient with 10s timeout
     - Parses with BeautifulSoup, extracts text from <p>, <li>, <h1>-<h4> tags
     - Strips navigation, footer, ads (remove elements with class containing
       "nav", "footer", "cookie", "ad", "sidebar", "menu")
     - Returns first 3000 characters of cleaned text
     - Returns empty string on any error (never raise — web is unreliable)

   async research_job(job_type: str, domain: str,
                      job_description: str) -> List[SearchResult]:
     Runs MULTIPLE focused search queries, one at a time with 0.5s delay:
     
     Query 1: f"what skills are needed for {job_type} {domain}"
     Query 2: f"tools used by {domain} professionals for {job_type}"
     Query 3: f"how to complete {job_description[:80]} step by step"
     Query 4: f"{domain} expert workflow {job_type} best practices"
     Query 5: f"AI agent tools for {domain} {job_type} 2025"
     
     Returns ALL results combined (deduplicated by URL).
     Log each query as it runs: "🔍 Searching: {query}"

2. In src/agentforge/research/result_parser.py:

   ResearchResultParser class:
   
   __init__(llm): takes a LangChain ChatModel for LLM parsing

   async parse(search_results: List[SearchResult],
               job: JobDefinition) -> ResearchResult:
     
     Step A — Prepare context:
       Concatenate all search result snippets and page content.
       Format as:
       "=== Source: {url} ===\n{title}\n{snippet}\n{full_content[:1000]}\n\n"
       Truncate total to 8000 chars if needed.
     
     Step B — LLM extraction call:
       system_prompt = """You are analyzing web research to understand what
       an AI agent needs to complete a specific job.
       
       Extract EXACTLY:
       1. required_skills: List of specific skills/knowledge areas needed
          (be concrete: "GAAP accounting" not just "finance")
       2. required_tool_types: List of TOOL TYPES needed
          (e.g. "web search", "spreadsheet", "PDF reader", "code executor",
           "database query", "email sender", "calculator", "chart generator")  
       3. expert_approach: List of 4-7 steps an expert would take to complete
          this job (be specific and actionable)
       4. domain_knowledge_summary: A 150-word paragraph of key domain
          knowledge the agent should have going in
       5. suggested_mcp_servers: Names of any specific MCP servers, npm packages,
          or Python libraries that would help (search results may mention these)
       6. confidence: Float 0-1. How complete is this research?
          1.0 = found detailed step-by-step guides
          0.5 = found general information
          0.2 = found little relevant content
       
       Respond ONLY with valid JSON matching the ResearchResult schema.
       Do not include any text outside the JSON."""
       
       user_prompt = f"""Job to research:
       Type: {job.job_type}
       Domain: {job.domain}  
       Description: {job.description}
       
       Web research findings:
       {context}
       
       Extract what this job requires."""
       
       Parse LLM JSON response into ResearchResult.
       Add search_queries_used and sources from the search_results.
     
     Step C — Validate and return:
       If required_skills is empty, set confidence = 0.1
       If len(required_tool_types) == 0, set required_tool_types = ["web_search"]
       Always ensure at least one expert_approach step exists.
       Return the ResearchResult.

   async refine(result: ResearchResult, 
                additional_query: str) -> ResearchResult:
     Run one more targeted search on a specific gap, merge findings.
     Used when coverage_score < 0.5 after initial research.

3. In src/agentforge/research/cache.py:

   ResearchCache class:
   - Stores ResearchResult objects as JSON files in .cache/research/
   - Cache key: sha256(f"{job_type}:{domain}")[:16]
   - TTL: RESEARCH_CACHE_TTL_DAYS days (default 7)
   
   async get(job_type: str, domain: str) -> Optional[ResearchResult]:
     Check file exists and is not expired. Return parsed result or None.
   
   async set(job_type: str, domain: str, result: ResearchResult):
     Write result as JSON to cache file with timestamp.
   
   async invalidate(job_type: str, domain: str):
     Delete the cache file.
   
   async stats() -> dict:
     Returns: {total_cached, oldest_entry, newest_entry, cache_size_kb}

4. In src/agentforge/research/mcp_discoverer.py:

   MCPDiscoverer class:
   - async find_servers_for_capability(capability: str) -> List[MCPServerSuggestion]
   
   MCPServerSuggestion:
     package_name: str       # e.g. "@modelcontextprotocol/server-brave-search"
     transport: str          # "stdio" or "http"
     capability_covered: str # what gap this fills
     install_command: str    # e.g. "npx -y @modelcontextprotocol/server-brave-search"
     notes: str              # any required API keys or config
   
   Searches: f"modelcontextprotocol {capability} mcp server npm"
   Parses results to find package names matching @modelcontextprotocol/* or
   other known MCP registries.
   Returns suggestions with install instructions.
   
   async save_suggestions(suggestions: List[MCPServerSuggestion],
                          job_id: str):
     Appends to logs/mcp_suggestions.jsonl so admins can review and install.

5. Unit tests in tests/unit/test_research.py:
   - test_search_returns_results: mock DDGS, verify SearchResult list returned
   - test_page_fetch_cleans_html: mock httpx, verify nav/footer stripped
   - test_parser_extracts_skills: mock LLM response, verify ResearchResult populated
   - test_cache_hit_skips_search: populate cache, verify search not called
   - test_cache_ttl_expired: set old timestamp, verify cache miss
   - test_empty_search_results_handled: verify graceful handling when web returns nothing
```

### Verify
```bash
uv run pytest tests/unit/test_research.py -v
```

---

## Step 1.3 — Tool Registry (Research-Aware)

### What you're building
A tool registry that maps `required_tool_types` from research (e.g. "chart generator") to actual callable tools. The trainer bridges the gap between what research says is needed and what tools actually exist.

### Claude Code Prompt

```
Build the Tool Registry in src/agentforge/tools/registry.py.
The key feature: tools are tagged with "tool_type" strings that match
what the ResearchResultParser extracts. This is how research findings
map to actual tools.

1. In src/agentforge/tools/registry.py:

   ToolRegistryEntry (Pydantic model):
     tool_id: str               # slug: "web_search", "calculator"
     name: str                  # human name
     description: str           # 2-3 sentences, what it does and when to use it
     tool_types: List[str]      # CRITICAL: matches what research extracts
       # e.g. ["web search", "real-time information", "fact lookup"]
       # These MUST align with the types the LLM extracts in result_parser.py
     capability_tags: List[str] # additional searchable tags
     source: str = "builtin"    # "builtin", "mcp://server-name"
     requires_auth: bool = False
     allowed_tenants: List[str] = ["*"]
     cost_tier: Literal["free","low","medium","high"] = "low"
     langchain_tool: Optional[Any] = None  # the actual callable tool
     class Config: arbitrary_types_allowed = True

   ToolRegistry class:
     __init__(): empty registry, builds index on first registration
     
     register(entry: ToolRegistryEntry): add to registry
     
     find_for_requirements(required_tool_types: List[str],
                           tenant_id: str) -> MatchResult:
       MatchResult:
         matched: List[ToolRegistryEntry]  # tools found for requirements
         unmatched: List[str]              # requirements with no tool found
         coverage: float                   # matched/total
       
       For each required_tool_type:
         - Exact string match against entry.tool_types first
         - If no exact match: fuzzy match (check if words overlap)
         - If still no match: add to unmatched list
       
       Deduplicate matched tools (one tool can cover multiple types).
       Return MatchResult.
     
     get_tool(tool_id: str) -> Optional[BaseTool]:
       Returns the actual LangChain BaseTool for calling.
     
     resolve_tools(tool_ids: List[str]) -> List[BaseTool]:
       Batch resolve tool_ids to LangChain tools.
     
     list_all_tool_types() -> List[str]:
       Returns all unique tool_types across all registered tools.
       Useful for debugging what the registry can cover.

2. In src/agentforge/tools/builtin/web_search.py:
   WebSearchTool using duckduckgo_search.
   tool_types: ["web search", "real-time information", "fact lookup",
               "current events", "research", "news"]
   Returns top 5 results as formatted string.

3. In src/agentforge/tools/builtin/page_reader.py:
   PageReaderTool — fetches and cleans a URL's content.
   tool_types: ["web page reader", "URL reader", "article reader",
               "document fetcher", "content extraction"]
   Input: url. Output: cleaned text (first 4000 chars).

4. In src/agentforge/tools/builtin/calculator.py:
   CalculatorTool — safe math via Python eval with restricted globals.
   tool_types: ["calculator", "math", "arithmetic", "computation",
               "financial calculation", "statistics"]
   Input: expression string. Output: numeric result + steps.

5. In src/agentforge/tools/builtin/file_ops.py:
   FileReaderTool and FileWriterTool.
   tool_types: ["file reader", "document reader", "PDF reader",
               "text reader", "file writer", "report generator",
               "document creator", "output writer"]

6. In src/agentforge/tools/builtin/database.py:
   DatabaseQueryTool (SQLite for MVP, SELECT-only).
   tool_types: ["database query", "SQL", "data retrieval",
               "structured data", "spreadsheet", "table lookup"]

7. In src/agentforge/tools/__init__.py:
   create_default_registry() -> ToolRegistry:
     Instantiates all 5 built-in tools, registers them with full metadata.
     Prints summary: "Tool Registry initialized: 5 tools, 30 tool types covered"

8. Tests in tests/unit/test_registry.py:
   - test_find_exact_match: "web search" requirement finds WebSearchTool
   - test_find_fuzzy_match: "internet search" finds WebSearchTool (word overlap)
   - test_unmatched_returned: "email sender" returns in unmatched list
   - test_coverage_calculation: 3 matched / 4 required = 0.75 coverage
   - test_resolve_returns_callable: resolved tools are BaseTool instances
```

### Verify
```bash
uv run pytest tests/unit/test_registry.py -v
```

---

## Step 1.4 — The Trainer Agent (Research → Worker Config)

### What you're building
The Trainer Agent — the central piece. It takes a job, searches the web to understand what's needed, maps findings to tools, handles gaps, then builds a complete `ContextPack` that configures the Worker Agent.

### Claude Code Prompt

```
Build the TrainerAgent in src/agentforge/core/trainer.py.
This is the most important class. Its ONLY job is:
  1. Research what the job requires (via web search)
  2. Map requirements to tools
  3. Build the Worker Agent's complete configuration

The Trainer never executes the job — it just prepares the Worker.

TrainerAgent class:

__init__(
  tool_registry: ToolRegistry,
  web_searcher: WebSearcher,
  result_parser: ResearchResultParser,
  research_cache: ResearchCache,
  mcp_discoverer: MCPDiscoverer,
  llm_planner,      # Claude Opus — for planning/reasoning
  llm_fast,         # GPT-4o-mini — for quick structured outputs
):

async train(job: JobDefinition) -> TrainingSession:

  TrainingSession:
    job_id: str
    research: ResearchResult
    context_pack: ContextPack
    match_result: MatchResult
    training_log: List[str]    # human-readable log of trainer's decisions
    duration_seconds: float

  The train() method must log EVERY decision it makes. Each step appends
  to training_log so the user can see exactly how the Worker was configured.

  ─── STAGE 1: RESEARCH ───────────────────────────────────────────────
  
  Step 1a — Check research cache:
    cached = await research_cache.get(job.job_type, job.domain)
    if cached and cached.confidence >= 0.7:
      log: "📦 Using cached research for {job.job_type}/{job.domain} 
            (confidence: {cached.confidence:.0%}, age: {age})"
      research = cached
    else:
      log: "🌐 No cache hit — starting web research for '{job.title}'"
      Go to Step 1b.
  
  Step 1b — Web research:
    log: "🔍 Researching: what does '{job.job_type}' in '{job.domain}' require?"
    search_results = await web_searcher.research_job(
      job.job_type, job.domain, job.description
    )
    log: f"📄 Found {len(search_results)} sources"
    
    research = await result_parser.parse(search_results, job)
    log: f"🧠 Research complete. Confidence: {research.confidence:.0%}"
    log: f"   Skills found: {', '.join(research.required_skills[:5])}"
    log: f"   Tool types needed: {', '.join(research.required_tool_types)}"
    
    If research.confidence < 0.4:
      log: "⚠️  Low confidence. Running refinement search..."
      refined_query = f"{job.domain} {job.job_type} detailed guide"
      research = await result_parser.refine(research, refined_query)
      log: f"   Refined confidence: {research.confidence:.0%}"
    
    await research_cache.set(job.job_type, job.domain, research)
    log: "💾 Research cached for future jobs"

  ─── STAGE 2: TOOL MATCHING ──────────────────────────────────────────
  
  Step 2a — Match research requirements to registry tools:
    match = tool_registry.find_for_requirements(
      research.required_tool_types, job.tenant_id
    )
    log: f"🔧 Tool matching: {len(match.matched)} tools found, 
          {len(match.unmatched)} gaps"
    for tool in match.matched:
      log: f"   ✅ '{tool.name}' covers: {tool.tool_types}"
    for gap in match.unmatched:
      log: f"   ❌ No tool for: '{gap}'"
  
  Step 2b — Handle gaps:
    if match.unmatched and settings.ENABLE_MCP_DISCOVERY:
      log: "🔭 Searching for MCP servers to fill gaps..."
      for gap in match.unmatched:
        suggestions = await mcp_discoverer.find_servers_for_capability(gap)
        if suggestions:
          log: f"   💡 For '{gap}': try '{suggestions[0].package_name}'"
          log: f"      Install: {suggestions[0].install_command}"
      await mcp_discoverer.save_suggestions(all_suggestions, job.job_id)
      log: "📝 MCP suggestions saved to logs/mcp_suggestions.jsonl"
  
  Step 2c — Select final tool set:
    Use match.matched as the tool set.
    Always ensure "web_search" is included (fallback for any unknown need).
    log: f"🎯 Final tools for worker: {[t.tool_id for t in final_tools]}"

  ─── STAGE 3: WORKER CONFIGURATION ──────────────────────────────────
  
  Step 3a — Build system prompt using the research:
    Call _build_system_prompt(job, research, final_tools) -> str
    log: f"📝 System prompt built ({estimated_tokens} tokens estimated)"
  
  Step 3b — Select model:
    _select_model(job, research) -> str:
      if job.priority == "critical" or research.confidence < 0.5:
        model = PLANNER_MODEL  # Use opus when uncertain
        log: "🤖 Using Opus (critical priority or low-confidence research)"
      else:
        model = DEFAULT_MODEL
        log: "🤖 Using Sonnet"
  
  Step 3c — Assess risk:
    _assess_risk(job, research) -> Literal["low","medium","high"]:
      "high" if any of: legal domain, contains "delete"/"send"/"payment",
                        finance domain + output_type=="action"
      "medium" if: hr domain, or research.confidence < 0.4
      "low" otherwise
    log: f"⚠️  Risk level: {risk}"
  
  Step 3d — Build and return TrainingSession:
    context_pack = ContextPack(
      system_prompt=system_prompt,
      role=_extract_role(research),
      goal=job.description,
      expert_approach=research.expert_approach,
      tools=[t.tool_id for t in final_tools],
      model=model,
      knowledge_context=research.domain_knowledge_summary,
      success_criteria=_derive_criteria(job, research),
      risk_level=risk,
      research_confidence=research.confidence,
    )
    log: f"✅ Training complete. Worker ready for: '{job.title}'"
    return TrainingSession(...)

_build_system_prompt(job, research, tools) -> str:
  Builds an XML-structured prompt:
  
  <role>
  You are a {_extract_role(research)} specializing in {job.domain}.
  </role>
  
  <goal>
  {job.description}
  </goal>
  
  <domain_knowledge>
  {research.domain_knowledge_summary}
  </domain_knowledge>
  
  <required_skills>
  You have been equipped with these skills relevant to this job:
  {numbered list of research.required_skills}
  </required_skills>
  
  <expert_approach>
  An expert would complete this job by following these steps:
  {numbered list of research.expert_approach}
  Follow this approach unless you discover a better path.
  </expert_approach>
  
  <available_tools>
  {formatted table: tool name | what it covers | when to use it}
  </available_tools>
  
  <constraints>
  {job.constraints joined with newlines}
  - Always use your tools to verify information before stating it as fact
  - If you encounter a task your tools cannot handle, say so clearly
  </constraints>

_extract_role(research) -> str:
  Use LLM (fast model) to generate a 1-line role title from research.required_skills.
  e.g. "Senior Financial Analyst and Reporting Specialist"

_derive_criteria(job, research) -> List[str]:
  Use LLM (fast model) to generate 3 success criteria based on job + research.

Tests in tests/unit/test_trainer.py:
  Mock all external calls (LLM, web searcher, cache).
  test_train_uses_cache_when_available
  test_train_runs_web_research_on_cache_miss
  test_train_logs_every_decision: verify training_log has entries for each stage
  test_train_includes_web_search_tool_always
  test_train_handles_zero_confidence_research: verify opus selected
  test_system_prompt_contains_all_sections: verify all XML tags present
  test_mcp_discoverer_called_on_gaps
```

### Verify
```bash
uv run pytest tests/unit/test_trainer.py -v
```

---

## Step 1.5 — Worker Agent & LangGraph Pipeline

### What you're building
The Worker Agent that executes jobs using its trainer-built `ContextPack`, and the LangGraph `StateGraph` that orchestrates the full flow: classify → research & train → execute → evaluate.

### Claude Code Prompt

```
Build the Worker Agent and the complete LangGraph pipeline.

1. In src/agentforge/core/worker.py:

   WorkerAgent class:

   async execute(job: JobDefinition, context_pack: ContextPack,
                 tool_registry: ToolRegistry) -> WorkerResult:
     
     WorkerResult:
       output: str
       tools_called: List[str]
       tool_call_count: int
       messages: List[dict]
       success: bool
       error: Optional[str]

     Step 1: Resolve tools from registry
       tools = tool_registry.resolve_tools(context_pack.tools)
       if not tools:
         return WorkerResult(error="No tools could be resolved")

     Step 2: Build the LangGraph ReAct agent
       Use langgraph.prebuilt.create_react_agent with:
         model: ChatAnthropic(model=context_pack.model)
         tools: resolved tools list
         state_modifier: context_pack.system_prompt (injected as system message)
     
     Step 3: Execute with timeout
       Use asyncio.wait_for(timeout=job.timeout_seconds)
       Input: {"messages": [HumanMessage(content=job.description)]}
     
     Step 4: Extract result
       Get final AIMessage from state["messages"]
       Extract tool names from all ToolMessage entries
       Return WorkerResult

     Step 5: Error handling
       TimeoutError → WorkerResult(error="Job timed out after {N}s", success=False)
       Any exception → WorkerResult(error=str(e), success=False)
       Log all errors with full traceback

2. In src/agentforge/core/meta_agent.py:

   MetaAgent — classifies raw job requests into JobDefinition.
   Uses GPT-4o-mini (cheap, fast, good at classification).
   
   async classify(raw_input: str, tenant_id: str) -> JobDefinition:
     Structured output call using pydantic_ai.
     System: "Parse this work request into a structured job definition.
              Infer job_type (snake_case), domain (single word), title (short),
              and output_type. Be decisive — never leave fields empty."
     Returns populated JobDefinition.

3. In src/agentforge/core/evaluator.py:

   JobEvaluator class:
   
   async evaluate(job: JobDefinition,
                  worker_result: WorkerResult,
                  research: ResearchResult) -> EvaluationResult:
     
     EvaluationResult:
       quality_score: float         # 0-1
       criteria_met: List[str]
       criteria_failed: List[str]
       feedback: str
       passed: bool                 # quality_score >= 0.65

     Uses LLM (fast model) to grade the output:
     "Given this job: {job.description}
      And this output: {worker_result.output[:1000]}
      The job required these skills: {research.required_skills}
      
      Score the output 0.0-1.0 on:
      - Completeness: did it address all parts of the job?
      - Accuracy: does it seem factually grounded?
      - Quality: is it useful and well-structured?
      
      Return JSON: {quality_score, criteria_met[], criteria_failed[], feedback}"

4. In src/agentforge/core/pipeline.py:

   AgentForgeState (TypedDict):
     messages: Annotated[list, add_messages]
     job: Optional[dict]
     training_session: Optional[dict]   # includes research + context_pack
     worker_result: Optional[dict]
     evaluation: Optional[dict]
     error: Optional[str]
     status: str
     tenant_id: str

   Build StateGraph with 5 nodes:

   "classify":
     MetaAgent.classify() → job dict
     status = "researching"

   "train":
     TrainerAgent.train() → TrainingSession
     Prints the training_log to console in real time:
       for line in session.training_log: print(f"  {line}")
     status = "running"

   "execute":
     WorkerAgent.execute() → WorkerResult
     status = "evaluating"

   "evaluate":
     JobEvaluator.evaluate() → EvaluationResult
     status = "success" or "failed"

   "handle_error":
     Log error, set status = "failed"

   Edges:
     START → "classify"
     "classify" → "train" (or "handle_error" on exception)
     "train" → "execute" (or "handle_error" on exception)
     "execute" → "evaluate"
     "evaluate" → END
     "handle_error" → END

   Compile with MemorySaver.

   async run_job(raw_input: str, tenant_id: str,
                 config: Optional[dict] = None) -> dict:
     Entry point. Returns final state dict.
     Prints status updates to console.

5. In src/agentforge/api/app.py:
   FastAPI with lifespan that builds all singletons:
     tool_registry = create_default_registry()
     web_searcher = WebSearcher()
     research_cache = ResearchCache()
     result_parser = ResearchResultParser(llm=fast_llm)
     mcp_discoverer = MCPDiscoverer()
     trainer = TrainerAgent(...)
     # Store all in app.state

6. In src/agentforge/api/routes/jobs.py:
   POST /jobs: {"description": str, "tenant_id": str} → runs pipeline
   Returns: JobResult with all fields including research_queries, skills_discovered

7. GET /health: {"status":"ok", "tools_registered": N, "cache_entries": N}

8. Integration test in tests/integration/test_pipeline.py:
   Mock LLM calls and web searches with realistic responses.
   test_full_pipeline: verify all 4 nodes run and state is populated
   test_training_log_populated: verify trainer decisions are logged
   test_research_findings_in_context_pack: verify skills from research in system prompt
   test_error_node_reached_on_failure: verify errors route correctly
```

### Verify
```bash
uv run pytest tests/integration/test_pipeline.py -v
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"description":"Analyze sentiment of customer reviews and summarize findings","tenant_id":"demo"}'
```

---

## Step 1.6 — Phase 1 Demo & README

### Claude Code Prompt

```
Create the Phase 1 demo script and full README.

1. scripts/demo_phase1.py — runs 4 diverse jobs to show research-first training:

   For EACH job, print a detailed before/after showing what research discovered:
   
   ┌─────────────────────────────────────────────────────┐
   │  JOB: {title}                                       │
   │  Domain: {domain}                                   │
   ├─────────────────────────────────────────────────────┤
   │  TRAINER RESEARCH LOG:                              │
   │  {training_log line by line}                        │
   ├─────────────────────────────────────────────────────┤  
   │  SKILLS DISCOVERED: {skills}                        │
   │  TOOL TYPES NEEDED: {tool_types}                    │
   │  TOOLS ASSIGNED:    {tools}                         │
   │  RESEARCH CONFIDENCE: {confidence}                  │
   ├─────────────────────────────────────────────────────┤
   │  WORKER OUTPUT PREVIEW:                             │
   │  {first 300 chars of output}                        │
   │  Quality Score: {score}                             │
   │  Cost: ${cost:.4f} | Duration: {duration:.1f}s      │
   └─────────────────────────────────────────────────────┘

   Test jobs:
   Job 1: "Research the top 5 machine learning frameworks and compare their
           strengths, weaknesses, and best use cases"
   
   Job 2: "Calculate the monthly payments on a $350,000 mortgage at 6.5%
           over 30 years, then show the full amortization for year 1"
   
   Job 3: "Find 3 recent news articles about quantum computing breakthroughs
           and summarize the key scientific advances"
   
   Job 4: "Write a Python function that sorts a list of employee records by
           salary descending, with unit tests"

2. README.md covering:
   - The core concept (3 paragraphs, no jargon)
   - How the research-first pipeline works (ASCII diagram)
   - Quickstart (5 commands: install, configure, run demo)
   - API usage with curl examples
   - How to interpret the trainer output (explain the training log)
   - What Phase 2 adds (MCP tool discovery, episodic memory, more)
   - FAQ: "Does it use the web for every job?" (answer: no, cached after first run)

3. Makefile:
   make install  → uv sync
   make dev      → uvicorn agentforge.api.app:app --reload --port 8000
   make test     → pytest tests/unit -v
   make demo     → python scripts/demo_phase1.py
   make lint     → ruff check src/
   make clean    → rm -rf .cache/ __pycache__/
```

---

# PHASE 2 — MEMORY & MCP
## Agents learn from past jobs and gain access to thousands of external tools

---

## Step 2.1 — Episodic Memory (Learning from Experience)

### Claude Code Prompt

```
Build the episodic memory system so the trainer learns from completed jobs.
After enough runs, the trainer recalls "last time I had a job like this,
these were the right tools" — and gets faster and smarter over time.

1. In src/agentforge/memory/episodic.py:

   ExperienceRecord (Pydantic model):
     experience_id: str
     job_type: str
     domain: str
     tenant_id: str
     job_description_summary: str  # first 200 chars
     research_queries_used: List[str]
     skills_discovered: List[str]
     tool_types_needed: List[str]
     tools_that_worked: List[str]   # tool_ids that were actually called
     quality_score: float
     research_confidence: float
     lessons: List[str]             # extracted by LLM post-evaluation
     approach_summary: str          # what approach worked
     duration_seconds: float
     cost_usd: float
     timestamp: datetime

   EpisodicMemory class (Qdrant-backed):
   
   async initialize(): creates "agentforge_episodes" collection in Qdrant.
     Vectors: OpenAI text-embedding-3-small, size=1536, COSINE distance.
   
   async remember(record: ExperienceRecord):
     Embed: f"{record.job_type} {record.domain} {record.approach_summary}"
     Payload: full ExperienceRecord dict
     Upsert to Qdrant.
   
   async recall(job: JobDefinition, top_k: int = 3,
                min_quality: float = 0.65) -> List[ExperienceRecord]:
     Query: f"{job.job_type} {job.domain} {job.description[:100]}"
     Filter: tenant_id == job.tenant_id AND quality_score >= min_quality
     Returns top_k most relevant past experiences.
   
   async extract_lessons(job: JobDefinition,
                         worker_result: WorkerResult,
                         evaluation: EvaluationResult,
                         llm) -> List[str]:
     LLM call: "Given this completed job and its evaluation, write 2-3
     specific lessons that would help an AI agent do better next time.
     Be concrete. Focus on: which tools were most useful, what approach
     worked, what to avoid."
     Returns list of lesson strings.

2. Update TrainerAgent.train() to use episodic memory:

   At the start of STAGE 1, before cache check:
   
   past_experiences = await episodic_memory.recall(job)
   if past_experiences:
     best = max(past_experiences, key=lambda x: x.quality_score)
     log: f"🧠 Found {len(past_experiences)} similar past jobs"
     log: f"   Best match: '{best.job_type}/{best.domain}' 
               (quality: {best.quality_score:.0%})"
     log: f"   Lessons: {'; '.join(best.lessons)}"
     
     # Inject lessons into the expert_approach of research result
     # If cache hit: merge past lessons into cached research
     # If cache miss: use past tool_types_needed to seed research queries
   
   If high-quality experience found (quality >= 0.8):
     log: "⚡ High-quality experience found — skipping web research"
     Reconstruct ResearchResult from the experience record.
     Set confidence = best.quality_score
     Skip web search entirely.

3. Update the "evaluate" pipeline node to call episodic_memory.remember():
   After evaluation completes:
   lessons = await episodic_memory.extract_lessons(job, worker_result, eval, llm)
   record = ExperienceRecord(
     job_type=job.job_type, domain=job.domain,
     tenant_id=job.tenant_id,
     job_description_summary=job.description[:200],
     research_queries_used=research.search_queries_used,
     skills_discovered=research.required_skills,
     tool_types_needed=research.required_tool_types,
     tools_that_worked=worker_result.tools_called,
     quality_score=evaluation.quality_score,
     research_confidence=research.confidence,
     lessons=lessons,
     approach_summary=f"Used {tools} with approach: {research.expert_approach[0]}",
     ...
   )
   await episodic_memory.remember(record)
   log: "💾 Experience stored for future jobs"

4. Add GET /memory/stats?tenant_id={id} API endpoint:
   Returns: {total_experiences, avg_quality, most_used_tools, domains_covered,
             research_cache_hits_pct (saved web searches %)}

5. Tests in tests/unit/test_memory.py:
   test_remember_and_recall_returns_relevant
   test_high_quality_experience_skips_web_search
   test_lessons_extracted_from_evaluation
   test_tenant_isolation_in_recall
```

---

## Step 2.2 — MCP Tool Integration

### Claude Code Prompt

```
Upgrade the tool registry to support MCP servers. When the trainer's
research discovers a tool type that no built-in tool covers, and
an MCP server IS installed, it gets picked up automatically.

1. In src/agentforge/tools/registry.py, add MCP support:

   async load_mcp_tools(mcp_configs: List[dict]):
     Uses langchain_mcp_adapters.client.MultiServerMCPClient
     For each discovered tool from MCP servers:
       - Infer tool_types from the tool's name and description using LLM:
         "Given this tool: {name} — {description}, list 3-5 tool_type strings
          that describe what it does (e.g. 'web search', 'database query').
          Return JSON array of strings only."
       - Register as ToolRegistryEntry with source="mcp://{server_name}"
     Log: "MCP tools loaded: {count} tools from {n} servers"

2. Update MCPDiscoverer to also CHECK if a suggested server is already
   installed (check if npx can find it):
   async is_installed(package_name: str) -> bool:
     Runs: subprocess.run(["npx", "--yes", "--dry-run", package_name])
     Returns True if no error.

3. Add docker-compose.yml with Qdrant and Redis services.

4. Update settings.py to support MCP_SERVERS config (JSON array in env var).

5. Update scripts/demo_phase1.py to show tool source in output:
   "✅ web_search (builtin) covers: web search, research"
   "✅ brave-search (mcp://brave) covers: web search, news, real-time"
```

---

# PHASE 3 — ENTERPRISE
## Temporal workflows, multi-tenancy, HITL, security

---

## Step 3.1 — Durable Workflows with Temporal

### Claude Code Prompt

```
Wrap the AgentForge pipeline in Temporal for enterprise durability.
Jobs must survive process crashes. High-risk jobs must pause for approval.

1. Create src/agentforge/core/workflows.py:

   @workflow.defn
   class AgentForgeWorkflow:
     
     @workflow.run
     async def run(self, params: WorkflowParams) -> WorkflowResult:
       self._approved = False
       self._denied = False
       
       # Activity: classify job
       job_dict = await workflow.execute_activity(
         classify_job_activity, params.raw_input, params.tenant_id,
         start_to_close_timeout=timedelta(seconds=30),
         retry_policy=RetryPolicy(max_attempts=3)
       )
       
       # Activity: trainer researches and builds ContextPack
       # (this may take 30-60s due to web searches)
       training_dict = await workflow.execute_activity(
         train_agent_activity, job_dict,
         start_to_close_timeout=timedelta(seconds=120),
         retry_policy=RetryPolicy(max_attempts=2)
       )
       
       # Gate: if high risk, pause and wait for human approval
       if training_dict["context_pack"]["risk_level"] == "high":
         await workflow.execute_activity(
           request_approval_activity, job_dict, training_dict,
           start_to_close_timeout=timedelta(seconds=10)
         )
         # Wait up to 24 hours for signal
         approved = await workflow.wait_condition(
           lambda: self._approved or self._denied,
           timeout=timedelta(hours=24)
         )
         if self._denied or not approved:
           return WorkflowResult(status="denied")
       
       # Activity: execute job
       result_dict = await workflow.execute_activity(
         execute_job_activity, job_dict, training_dict,
         start_to_close_timeout=timedelta(seconds=job_dict.get("timeout_seconds", 300)),
         retry_policy=RetryPolicy(max_attempts=1)
       )
       
       # Activity: evaluate
       eval_dict = await workflow.execute_activity(
         evaluate_result_activity, job_dict, result_dict, training_dict,
         start_to_close_timeout=timedelta(seconds=30)
       )
       
       return WorkflowResult(status="success", result=result_dict, evaluation=eval_dict)
     
     @workflow.signal
     def approve(self, reviewer_id: str): self._approved = True
     
     @workflow.signal  
     def deny(self, reviewer_id: str, reason: str): self._denied = True
     
     @workflow.query
     def status(self) -> str:
       if self._approved: return "approved"
       if self._denied: return "denied"
       return "pending_approval"

2. Create src/agentforge/core/activities.py with all @activity.defn functions.
   Each imports and calls the core classes.

3. Create src/agentforge/core/worker_process.py — the Temporal worker runner.

4. Update docker-compose.yml to add Temporal server and UI.

5. Update API: POST /jobs now returns workflow_id immediately.
   GET /jobs/{workflow_id}/status returns current state.
   POST /jobs/{workflow_id}/approve sends signal.
```

---

## Step 3.2 — Multi-Tenancy, Security & Production Hardening

### Claude Code Prompt

```
Add enterprise security, multi-tenancy, and production hardening.

1. TenantRegistry with per-tenant configs:
   allowed_domains, allowed_tool_types, max_cost_per_day_usd,
   requires_approval_for (list of job_types), isolation_mode

2. RBAC middleware that validates X-Tenant-ID on every request.

3. Immutable audit log (JSONL append-only files) for:
   every job submission, every tool call, every approval event.

4. Per-tenant cost caps: reject jobs when daily limit reached.
   Track costs via audit log, expose GET /tenants/{id}/usage.

5. InputSanitizer: detect and neutralize prompt injection in job descriptions.
   Wrap user input in XML delimiters before injecting into prompts.

6. Rate limiting: 10 job submissions/minute per tenant.

7. Production docker-compose.prod.yml with resource limits, health checks,
   restart policies, and no exposed internal ports.

8. Full docs/security.md covering: tenant setup, secrets management,
   audit log queries, incident response.
```

---

# PHASE 4 — OBSERVABILITY & PRODUCTION
## Full tracing, evals, performance, and the final demo

---

## Step 4.1 — Langfuse Observability

### Claude Code Prompt

```
Add Langfuse tracing throughout the entire pipeline.

1. Create ObservabilityManager with Langfuse client singleton.

2. Instrument every LLM call and tool call with Langfuse callbacks.

3. Create one trace per job containing:
   - Span for each pipeline node (classify, train, execute, evaluate)
   - Sub-spans for each web search query during training
   - Sub-spans for each tool call during worker execution
   - Token counts and cost at every LLM call
   - Final quality_score as a Langfuse score event
   - Tags: job_type, domain, tenant_id, used_cache (T/F), used_episodic (T/F)

4. Add cost tracking per model using pricing table in config/models.yaml.

5. GET /metrics endpoint: last-7-days summary per tenant.
   Include: avg research time, avg execution time, cache hit rate,
   episodic memory hit rate, avg quality score, total cost.
```

---

## Step 4.2 — Evaluation Suite & Final Demo

### Claude Code Prompt

```
Build the evaluation suite and the comprehensive final demo.

1. tests/evals/ — LLM-as-judge evaluators for:
   - Output quality (completeness, accuracy, usefulness)
   - Tool selection accuracy (did research find the right tools?)
   - Research quality (did web search find relevant information?)
   
2. Test datasets in tests/evals/datasets/ covering 5 diverse job types.

3. scripts/benchmark.py — runs all evals, compares to baseline, fails on regression.

4. scripts/demo_final.py — 5 diverse jobs showing the full system:

   Job 1: "maritime insurance claim processing" (zero-shot — never seen this domain)
     Show: trainer researches from scratch, discovers niche domain tools
   
   Job 2: Run same job again immediately
     Show: cache hit — no web search, instant training from cached research
   
   Job 3: Run 10 jobs in the "finance" domain over time
     Show: episodic memory kicks in on job 4+, quality improves
   
   Job 4: "delete all records older than 2019 from the customer database"
     Show: high-risk detection, approval gate triggered, human approves
   
   Job 5: An entirely novel job type the system has never seen
     Show: low-confidence research → refined search → opus model selected
           → MCP gap suggestions logged for admin

5. Final README update with:
   - Full architecture diagram
   - Research caching explanation (how it saves cost over time)
   - Episodic memory explanation (how quality improves over time)
   - Performance numbers from benchmark
   - Complete API reference
```

---

# ALL PROMPTS AT A GLANCE

| # | What It Builds | Run After |
|---|---------------|-----------|
| 1.1 | Schemas + folder structure | project init |
| 1.2 | WebSearcher + ResultParser + ResearchCache | 1.1 |
| 1.3 | Tool Registry with tool_type matching | 1.2 |
| 1.4 | TrainerAgent (research → worker config) | 1.3 |
| 1.5 | WorkerAgent + LangGraph pipeline + API | 1.4 |
| 1.6 | Phase 1 demo + README | 1.5 |
| 2.1 | Episodic memory (learn from past jobs) | 1.6 |
| 2.2 | MCP tool integration | 2.1 |
| 3.1 | Temporal durable workflows + approval gate | 2.2 |
| 3.2 | Multi-tenancy, security, audit logging | 3.1 |
| 4.1 | Langfuse observability + cost tracking | 3.2 |
| 4.2 | Eval suite + final demo | 4.1 |

---

# The Research-First Flow (Diagram)

```
 RAW JOB REQUEST
 "Audit our GDPR data practices"
        │
        ▼
 ┌─────────────┐
 │ META-AGENT  │  Classifies: job_type="compliance_audit", domain="legal"
 └──────┬──────┘
        │
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                     TRAINER AGENT                           │
 │                                                             │
 │  1. Check episodic memory ──► Similar job found (quality    │
 │     for "compliance_audit"      0.85)? → skip web search   │
 │     in "legal"                  No match? → go to step 2   │
 │                                                             │
 │  2. Check research cache ───► Cache hit (< 7 days old)?    │
 │     for this job_type/domain    Yes? → use cached research  │
 │                                 No? → go to step 3         │
 │                                                             │
 │  3. WEB SEARCH (if needed)                                  │
 │     ├─ "what skills does compliance_audit in legal need"    │
 │     ├─ "tools used for GDPR compliance auditing"           │
 │     ├─ "how to audit GDPR data practices step by step"     │
 │     ├─ "legal compliance expert workflow best practices"    │
 │     └─ "AI agent tools for legal compliance 2025"          │
 │                │                                            │
 │                ▼                                            │
 │     ResultParser extracts:                                  │
 │     skills: ["GDPR knowledge","data mapping","legal        │
 │              analysis","regulatory framework review"]       │
 │     tool_types: ["document reader","web search",           │
 │                  "file writer","calculator"]               │
 │     expert_approach: [4 steps an auditor takes]            │
 │                                                             │
 │  4. TOOL MATCHING                                           │
 │     "document reader"  ──► FileReaderTool ✅               │
 │     "web search"       ──► WebSearchTool ✅                │
 │     "file writer"      ──► FileWriterTool ✅               │
 │     "calculator"       ──► CalculatorTool ✅               │
 │     coverage: 100%                                          │
 │                                                             │
 │  5. BUILD WORKER CONFIGURATION                              │
 │     role: "Compliance & Regulatory Audit Specialist"       │
 │     system prompt: role + domain knowledge + expert steps  │
 │     tools assigned: [file_reader, web_search, file_writer] │
 │     model: claude-sonnet (high confidence research)        │
 │     risk_level: HIGH → approval required                   │
 └──────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ APPROVAL GATE   │  (risk=high → pause)
                   │ Notifies human  │
                   │ Waits for OK    │
                   └───────┬─────────┘
                           │ approved
                           ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                      WORKER AGENT                           │
 │  Equipped with: compliance specialist role + GDPR knowledge │
 │                 + 4-step expert approach + 3 tools          │
 │                                                             │
 │  Executes the job using exactly what the trainer gave it.   │
 └──────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   EVALUATOR     │  quality_score: 0.88
                   └───────┬─────────┘
                           │
                           ▼
                   ┌─────────────────┐
                   │ EPISODIC MEMORY │  Stores experience.
                   │    STORAGE      │  Next GDPR audit:
                   │                 │  skips web search.
                   └─────────────────┘
```

---

# Key Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...            # for embeddings + router model

DEFAULT_MODEL=claude-sonnet-4-5-20251001
PLANNER_MODEL=claude-opus-4-5-20251101
ROUTER_MODEL=gpt-4o-mini

QDRANT_URL=http://localhost:6333

LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

RESEARCH_CACHE_TTL_DAYS=7        # how long to cache web research
MAX_SEARCH_QUERIES_PER_JOB=5     # web searches per training session
MAX_PAGES_TO_FETCH=3             # pages to read per search
ENABLE_MCP_DISCOVERY=true        # search for MCP servers for gaps
```
