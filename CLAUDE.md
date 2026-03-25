# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentForge is a **web-research-first AI agent training system**. The core mechanic: when a job arrives, a Trainer Agent searches the web to discover what skills and tools the job requires, then assembles a Worker Agent from scratch with exactly those capabilities. Nothing is hardcoded — the trainer discovers everything at runtime, caches results, and learns from past executions.

**Current state:** Specification only. `AgentForge_WebResearch_Plan.md` is the authoritative implementation blueprint with 15 sequential Claude Code prompts. No source code exists yet.

## Commands

```bash
# Dependencies (Python 3.12 via uv)
uv sync
uv python install 3.12

# Verify schemas work
uv run python -c "from agentforge.schemas.research import ResearchResult; print('OK')"

# Run tests
uv run pytest tests/unit -v
uv run pytest tests/unit/test_research.py -v     # research engine only
uv run pytest tests/unit/test_registry.py -v     # tool registry only
uv run pytest tests/unit/test_trainer.py -v      # trainer agent only
uv run pytest tests/integration/test_pipeline.py -v
uv run pytest tests/evals/ -m "not slow"

# Development server
uvicorn agentforge.api.app:app --reload --port 8000

# Demo & utilities
python scripts/demo_phase1.py
python scripts/benchmark.py

# Lint & format
ruff check src/
ruff format src/

# Makefile shortcuts
make install   # uv sync
make dev       # uvicorn with reload on port 8000
make test      # pytest tests/unit -v
make demo      # python scripts/demo_phase1.py
make lint      # ruff check src/
make clean     # rm -rf .cache/ __pycache__/
```

## Architecture

### The Research-First Pipeline

```
Request → [classify] → [train] → [execute] → [evaluate] → Result
```

All 5 LangGraph `StateGraph` nodes live in `core/pipeline.py`. State type is `AgentForgeState` (TypedDict).

1. **classify** — `MetaAgent` (`core/meta_agent.py`) parses raw input into a structured `JobDefinition` using GPT-4o-mini via pydantic-ai structured output
2. **train** — `TrainerAgent` (`core/trainer.py`) is the central piece:
   - Checks `EpisodicMemory` for past similar jobs first (skips web research if quality ≥ 0.8 match found)
   - Checks `ResearchCache` (7-day TTL, keyed by `sha256(job_type:domain)[:16]`)
   - If cache miss: `WebSearcher` runs 5 focused queries via DuckDuckGo, `ResearchResultParser` extracts `ResearchResult` via LLM
   - Maps `ResearchResult.required_tool_types` → actual tools via `ToolRegistry.find_for_requirements()`
   - For unmatched tool types: `MCPDiscoverer` suggests MCP servers (logged to `logs/mcp_suggestions.jsonl`)
   - Builds XML-structured system prompt from research findings, selects model (Opus for critical/low-confidence, Sonnet otherwise), assesses risk
   - Returns `TrainingSession` with full `training_log` of every decision
3. **execute** — `WorkerAgent` (`core/worker.py`) creates a LangGraph ReAct agent with `create_react_agent`, executes with `asyncio.wait_for` timeout
4. **evaluate** — `JobEvaluator` (`core/evaluator.py`) scores output, then `EpisodicMemory.remember()` stores the experience for future jobs

### Key subsystems

- **`research/`** — `WebSearcher` (DuckDuckGo + httpx page fetching), `ResearchResultParser` (LLM extracts `ResearchResult` from web content), `ResearchCache` (file-based JSON cache in `.cache/research/`), `MCPDiscoverer`
- **`tools/registry.py`** — `ToolRegistry` maps `required_tool_types` strings from research to actual `BaseTool` instances. Matching is exact-first, then fuzzy (word overlap). Phase 2 adds `load_mcp_tools()` via `langchain_mcp_adapters`
- **`tools/builtin/`** — 5 tools: `web_search` (DuckDuckGo), `page_reader` (httpx+BeautifulSoup), `calculator` (safe eval), `file_ops` (read/write), `database` (SQLite SELECT-only). Each has `tool_types` list that must align with what `ResearchResultParser` extracts
- **`memory/episodic.py`** — Qdrant-backed, collection `agentforge_episodes`, OpenAI embeddings (text-embedding-3-small). Recalled experiences with quality ≥ 0.8 skip web research entirely
- **`prompts/builder.py`** — builds XML-structured system prompts with `<role>`, `<goal>`, `<domain_knowledge>`, `<required_skills>`, `<expert_approach>`, `<available_tools>`, `<constraints>` sections
- **`api/app.py`** — FastAPI with lifespan initializing all singletons into `app.state`. Routes: `POST /jobs`, `GET /health`, `GET /memory/stats`

### Schemas (`schemas/`)

- `JobDefinition` — input: `job_type`, `domain`, `output_type`, `constraints`, `tenant_id`
- `ResearchResult` — trainer output from web: `required_skills`, `required_tool_types`, `expert_approach`, `domain_knowledge_summary`, `confidence` (0-1)
- `ContextPack` — worker config: `system_prompt`, `tools` list, `model`, `knowledge_context`, `risk_level`, `research_confidence`
- `TrainingSession` — wraps `ResearchResult` + `ContextPack` + `training_log` + `match_result`
- `JobResult` — API response including `research_queries`, `skills_discovered`, `cost_usd`

### Technology stack

| Layer | Tech |
|---|---|
| Language | Python 3.12 (`uv`) |
| Orchestration | LangGraph 0.2+ (`StateGraph` + `create_react_agent`) |
| LLMs | Anthropic Claude opus/sonnet, OpenAI GPT-4o-mini |
| Web search | `duckduckgo-search` (no API key), `httpx`, `beautifulsoup4` |
| Vector DB | Qdrant (`agentforge_episodes` collection) |
| Embeddings | OpenAI text-embedding-3-small |
| MCP tools | `langchain-mcp-adapters` (Phase 2) |
| Workflow durability | Temporal (Phase 3) |
| API | FastAPI + uvicorn |
| Validation | Pydantic v2 + pydantic-ai |
| Observability | Langfuse (Phase 4) |
| Testing | pytest + pytest-asyncio |
| Linting | ruff |

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
DEFAULT_MODEL=claude-sonnet-4-6          # worker agent default
PLANNER_MODEL=claude-opus-4-6            # trainer planning, critical jobs
ROUTER_MODEL=gpt-4o-mini                 # classification, fast structured outputs
QDRANT_URL=http://localhost:6333
RESEARCH_CACHE_TTL_DAYS=7
MAX_SEARCH_QUERIES_PER_JOB=5
MAX_PAGES_TO_FETCH=3
ENABLE_MCP_DISCOVERY=true
LANGFUSE_PUBLIC_KEY=pk-lf-...            # Phase 4
LANGFUSE_SECRET_KEY=sk-lf-...            # Phase 4
TEMPORAL_HOST=localhost:7233             # Phase 3
```

## Implementation Phases

| Phase | Focus |
|---|---|
| 1 | Research engine, tool registry, trainer+worker+evaluator, LangGraph pipeline, FastAPI |
| 2 | Episodic memory (Qdrant), MCP tool integration, docker-compose |
| 3 | Temporal durable workflows, multi-tenancy RBAC, human-in-the-loop approvals |
| 4 | Langfuse observability, evaluation suite, security hardening |

Consult `AgentForge_WebResearch_Plan.md` for exact field definitions, per-step exit criteria, and the 15 sequential implementation prompts.
