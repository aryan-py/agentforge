# AgentForge

**Web-research-first AI agent config generator.**

Describe your agent's job in plain English. AgentForge searches the web across 10 targeted queries to figure out what skills, tools, and MCP servers it needs — then generates a ready-to-use ZIP of config files you can drop straight into your project.

No templates. No hardcoded assumptions. Every config is researched from scratch.

---

## What it does

You type:
> *"Monitor a GitHub repository for new issues, search the web for similar bugs, and post a daily Slack digest with recommended fixes."*

AgentForge:
1. Runs **10 DuckDuckGo searches** — skills required, tools used by professionals, MCP servers, npm packages, APIs, best practices
2. Fetches and reads the top pages from each search
3. Uses GPT-4o-mini to extract a complete agent profile
4. Asks you a few questions (framework, MCP server preferences, constraints)
5. Generates and packages **5 config files** for download

---

## Generated files

| File | What it is |
|------|------------|
| `mcp_config.json` / `mcp_config.py` | MCP server configuration for your chosen framework |
| `system_prompt.txt` | Pre-filled system prompt with all discovered skills, domain knowledge, and expert approach |
| `skills.yaml` | Structured capability profile — skills, tool types, confidence score |
| `requirements.txt` | Python dependencies inferred from the job domain |
| `README.md` | Setup instructions with API key links and step-by-step guide |

---

## Supported frameworks

| Framework | Config format |
|-----------|--------------|
| Claude Desktop | JSON (`claude_desktop_config.json`) |
| Cursor | JSON (`.cursor/mcp.json`) |
| Windsurf | JSON (`.windsurf/mcp_config.json`) |
| LangGraph | Python dict (`mcp_config.py`) |
| LangChain | Python dict (`mcp_config.py`) |
| CrewAI | Python dict (`mcp_config.py`) |
| Custom Python | Python dict (`mcp_config.py`) |

---

## Quickstart

### Requirements
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- Node.js ≥ 18 (for running MCP servers via `npx`)

### 1. Clone and install

```bash
git clone git@github.com:aryan-py/agentforge.git
cd agentforge
uv sync
```

### 2. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and set your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-key-here
```

That's the only required field. Everything else has sensible defaults.

### 3. Run the dashboard

```bash
make dashboard
# or
uv run streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

### 4. Generate a config

1. Click **Config Generator** in the sidebar
2. Describe your agent's job
3. Wait ~60 seconds for web research to complete
4. Choose your framework and MCP servers
5. Download the ZIP

---

## Dashboard pages

### Home (`/`)
Stats and history — total configs generated, research cache entries, skills discovered. Shows every past config with its domain, framework, confidence score, and settings.

### Config Generator
The main feature. 4-step flow:

```
Describe → Research → Configure → Review & Download
```

**Step 1 — Describe**
Enter your agent's job description. Be specific: mention the domain, integrations, and output format.

**Step 2 — Research**
AgentForge runs 10 web searches in the background. A progress bar shows what's happening. Results are cached for 7 days — the same job type + domain combination won't be searched again until the cache expires.

**Step 3 — Configure**
Review what the research found:
- Skills (shown as blue chips)
- Tool types needed (shown as green chips)
- Expert approach (step-by-step)
- Sources consulted

Then choose:
- Target framework
- Which MCP servers to include (free vs. API-key-required)
- Custom role title (optional)
- Additional constraints (optional)

**Step 4 — Review & Download**
Preview all 5 generated files in tabs. Download as a single ZIP.

---

## How the research engine works

```
Job description
      │
      ▼
 MetaAgent classifies → job_type + domain
      │
      ▼
 ResearchCache check → hit? return cached result
      │ miss
      ▼
 WebSearcher runs 10 queries:
   • "what skills are required for {job_type} in {domain}"
   • "best tools and software for {job_type} {domain} professionals"
   • "Python libraries and packages for {domain} {job_type} automation"
   • "MCP server tools for {domain} {job_type} AI agent"
   • "npm MCP packages {domain} automation tools"
   • "how to automate {description} step by step"
   • "{domain} expert workflow best practices tutorial"
   • "AI agent {domain} {job_type} tools plugins"
   • "{domain} APIs and integrations 2024 2025"
   • + 1 more targeted query
      │
      ▼
 Top 4 pages fetched per query (5000 chars each)
      │
      ▼
 ResearchResultParser (GPT-4o-mini, 14k context)
   extracts: skills, tool_types, expert_approach,
             domain_knowledge, suggested_packages, confidence
      │
      ▼ confidence < 50%?
 Refinement pass (2 more targeted searches + re-parse)
      │
      ▼
 MCPCatalog.find_servers_for_tool_types()
   matches tool types → curated MCP server specs
      │
      ▼
 ResearchResult cached for 7 days
```

---

## MCP server catalog

AgentForge includes a curated catalog of 13 MCP servers matched to common tool types:

| Server | Tool types | Key required? |
|--------|------------|---------------|
| Filesystem | file read/write | No |
| Fetch (web) | web scraper, page reader | No |
| Memory | knowledge base, note-taking | No |
| Git | version control, code review | No |
| SQLite | database query, SQL | No |
| GitHub | GitHub API, issue tracking | Yes (GitHub token) |
| Slack | messaging, notifications | Yes (Slack token) |
| PostgreSQL | database query | Yes (connection string) |
| Brave Search | web search | Yes (Brave API key) |
| Google Maps | geolocation | Yes (Google API key) |
| Puppeteer | browser automation | No (needs Chrome) |
| Everything Search | file search | No (Windows only) |
| AWS KB Retrieval | vector search | Yes (AWS credentials) |

Servers are automatically suggested based on the tool types the research discovers. You choose which ones to include before downloading.

---

## Project structure

```
agentforge/
├── dashboard/
│   ├── app.py                    # Home page — stats and history
│   └── pages/
│       └── 1_Config_Generator.py # Main feature
├── src/agentforge/
│   ├── config/settings.py        # Pydantic settings (reads .env)
│   ├── config_generator/
│   │   ├── generator.py          # Generates all 5 config files
│   │   ├── mcp_catalog.py        # 13 curated MCP server specs
│   │   └── packager.py           # Builds the ZIP archive
│   ├── core/
│   │   ├── meta_agent.py         # Classifies job description
│   │   ├── trainer.py            # Orchestrates research + tool matching
│   │   ├── worker.py             # LangGraph ReAct agent executor
│   │   ├── evaluator.py          # Scores output quality
│   │   └── pipeline.py           # LangGraph StateGraph (5 nodes)
│   ├── research/
│   │   ├── web_searcher.py       # DuckDuckGo + httpx page fetcher
│   │   ├── result_parser.py      # LLM extracts ResearchResult from web
│   │   ├── cache.py              # File-based JSON cache (.cache/research/)
│   │   └── mcp_discoverer.py     # Suggests MCP servers for tool gaps
│   ├── schemas/
│   │   ├── job.py                # JobDefinition, JobResult
│   │   ├── research.py           # ResearchResult, SearchResult
│   │   └── context_pack.py       # ContextPack (worker config)
│   └── tools/
│       ├── registry.py           # Maps tool_type strings → LangChain tools
│       └── builtin/              # 6 built-in tools
├── tests/                        # pytest unit + integration tests
├── .env.example                  # Copy to .env and fill in keys
├── pyproject.toml                # Dependencies (uv)
└── Makefile                      # Common commands
```

---

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | **Yes** | — | Used for all LLM calls |
| `ANTHROPIC_API_KEY` | No | — | Optional, not used by default |
| `DEFAULT_MODEL` | No | `gpt-4o-mini` | Worker agent model |
| `PLANNER_MODEL` | No | `gpt-4o-mini` | Trainer planning model |
| `ROUTER_MODEL` | No | `gpt-4o-mini` | Classification model |
| `RESEARCH_CACHE_TTL_DAYS` | No | `7` | Days before cached research expires |
| `QDRANT_URL` | No | `localhost:6333` | Vector DB for episodic memory (Phase 2) |
| `TEMPORAL_HOST` | No | `localhost:7233` | Durable workflows (Phase 3) |

---

## Development

```bash
make install        # install all dependencies
make dashboard      # run Streamlit dashboard
make dev            # run FastAPI server (port 8000)
make test           # run unit tests
make lint           # ruff check
make format         # ruff format
make clean          # remove .cache/ and __pycache__/
```

Run a specific test:

```bash
uv run pytest tests/unit/test_research.py -v
uv run pytest tests/unit/test_registry.py -v
```

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12 |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| LLM | OpenAI GPT-4o-mini |
| Web search | DuckDuckGo (no API key) |
| Page fetching | httpx + BeautifulSoup4 |
| Orchestration | LangGraph |
| Validation | Pydantic v2 |
| Dashboard | Streamlit |
| API | FastAPI |
| Testing | pytest + pytest-asyncio |
| Linting | ruff |

---

## License

MIT
