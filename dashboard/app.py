"""AgentForge Dashboard — Home with stats and history."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env into os.environ before any client is instantiated
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

st.set_page_config(
    page_title="AgentForge",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stat-label { font-size: 0.8em; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
.skill-chip {
    display: inline-block; background: #1e3a5f; color: #7eb8f7;
    border-radius: 12px; padding: 2px 10px; margin: 2px; font-size: 0.78em;
}
.tool-chip {
    display: inline-block; background: #1e3a2a; color: #5ecf8a;
    border-radius: 12px; padding: 2px 10px; margin: 2px; font-size: 0.78em;
}
</style>
""", unsafe_allow_html=True)

# ── Data helpers ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_CACHE_DIR = _ROOT / ".cache" / "research"
_HIST_FILE = _ROOT / "logs" / "config_history.jsonl"


def _load_history():
    if not _HIST_FILE.exists():
        return []
    entries = []
    for line in _HIST_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return list(reversed(entries))  # newest first


def _load_cache_entries():
    if not _CACHE_DIR.exists():
        return []
    entries = []
    for f in sorted(_CACHE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            entries.append(data)
        except Exception:
            pass
    return entries


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 AgentForge")
st.caption(
    "Web-research-first AI agent config generator. "
    "Describe your job → get skills, tools, MCP servers, and ready-to-use config files."
)
st.divider()

history = _load_history()
cache_entries = _load_cache_entries()

# ── Top-level stats ───────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Configs generated", len(history))
c2.metric("Research cache entries", len(cache_entries))

total_skills = sum(e.get("skills_count", 0) for e in history)
total_tools = sum(e.get("tools_count", 0) for e in history)
avg_conf = (
    sum(e.get("confidence", 0) for e in history) / len(history) if history else 0
)
c3.metric("Total skills discovered", total_skills)
c4.metric("Total tool types found", total_tools)
c5.metric("Avg research confidence", f"{avg_conf:.0%}" if history else "—")

st.divider()

# ── Main content ──────────────────────────────────────────────────────────────
tab_history, tab_cache, tab_about = st.tabs(
    ["📋 Config History", "🔬 Research Cache", "ℹ️ About"]
)

# ── Tab 1: Config History ─────────────────────────────────────────────────────
with tab_history:
    if not history:
        st.info("No configs generated yet. Use the **Config Generator** in the sidebar to get started.")
    else:
        st.markdown(f"### Last {min(len(history), 50)} generated configs")

        for i, entry in enumerate(history[:50]):
            generated_at = entry.get("generated_at", "")
            try:
                ts = datetime.fromisoformat(generated_at).strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                ts = generated_at[:19]

            conf = entry.get("confidence", 0)
            conf_color = "#34c78a" if conf >= 0.7 else "#f7a94f" if conf >= 0.4 else "#e05c5c"

            with st.expander(
                f"**{entry.get('domain', '').title()} / {entry.get('job_type', '')}** "
                f"· {entry.get('framework', '')} · {ts}",
                expanded=(i == 0),
            ):
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Framework", entry.get("framework", "—"))
                col_b.metric("Skills", entry.get("skills_count", 0))
                col_c.metric("Tool types", entry.get("tools_count", 0))
                col_d.metric("MCP servers", entry.get("mcp_servers_count", 0))

                st.markdown(
                    f"<span style='color:{conf_color}'>● Confidence: {conf:.0%}</span>",
                    unsafe_allow_html=True,
                )

                if entry.get("description"):
                    st.markdown(f"**Job description:** {entry['description']}")

                if entry.get("zip_name"):
                    st.code(entry["zip_name"], language=None)

# ── Tab 2: Research Cache ─────────────────────────────────────────────────────
with tab_cache:
    if not cache_entries:
        st.info("Research cache is empty. Run a config generation to populate it.")
    else:
        # Cache-level stats
        total_kb = sum(
            (_CACHE_DIR / f"{_k}.json").stat().st_size
            for _k in [
                __import__("hashlib").sha256(f"{e.get('job_type','')}:{e.get('domain','')}".encode()).hexdigest()[:16]
                for e in cache_entries
            ]
            if (_CACHE_DIR / f"{_k}.json").exists()
        ) / 1024

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Cached research entries", len(cache_entries))
        sc2.metric("Cache size", f"{total_kb:.1f} KB")
        sc3.metric("TTL", "7 days")

        st.divider()
        st.markdown(f"### {len(cache_entries)} cached research results")

        for entry in cache_entries:
            cached_at = entry.get("_cached_at", entry.get("researched_at", ""))
            try:
                ts = datetime.fromisoformat(cached_at).strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                ts = cached_at[:19]

            conf = float(entry.get("confidence", 0))
            conf_color = "#34c78a" if conf >= 0.7 else "#f7a94f" if conf >= 0.4 else "#e05c5c"

            with st.expander(
                f"**{entry.get('domain', '').title()}** / `{entry.get('job_type', '')}` · cached {ts}"
            ):
                ca, cb, cc = st.columns(3)
                ca.metric("Skills", len(entry.get("required_skills", [])))
                cb.metric("Tool types", len(entry.get("required_tool_types", [])))
                cc.markdown(
                    f"<span style='color:{conf_color}'>● Confidence: {conf:.0%}</span>",
                    unsafe_allow_html=True,
                )

                skills = entry.get("required_skills", [])
                if skills:
                    st.markdown("**Skills:**")
                    chips = " ".join(f"<span class='skill-chip'>{s}</span>" for s in skills)
                    st.markdown(chips, unsafe_allow_html=True)

                tools = entry.get("required_tool_types", [])
                if tools:
                    st.markdown("**Tool types:**")
                    chips = " ".join(f"<span class='tool-chip'>{t}</span>" for t in tools)
                    st.markdown(chips, unsafe_allow_html=True)

                pkgs = entry.get("suggested_mcp_servers", [])
                if pkgs:
                    st.markdown("**Suggested packages:**")
                    st.markdown(" · ".join(f"`{p}`" for p in pkgs))

                sources = entry.get("sources", [])
                if sources:
                    st.markdown(f"**{len(sources)} sources consulted**")

                if entry.get("domain_knowledge_summary"):
                    st.markdown("**Domain knowledge:**")
                    st.markdown(entry["domain_knowledge_summary"])

# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
### How AgentForge works

1. **Describe** your agent's job in plain English
2. **Research** — AgentForge runs 10 targeted web searches across skills, tools, MCP servers, APIs, and best practices
3. **Configure** — pick your framework, select MCP servers, add constraints
4. **Download** — get a ZIP with 5 ready-to-use files:

| File | What it is |
|------|------------|
| `mcp_config` | MCP server config for your chosen framework |
| `system_prompt.txt` | Pre-filled system prompt with all discovered skills |
| `skills.yaml` | Structured capability profile |
| `requirements.txt` | Python dependencies |
| `README.md` | Setup instructions with API key links |

### Supported frameworks
- Claude Desktop, Cursor, Windsurf — JSON config
- LangGraph, LangChain, CrewAI, Custom Python — Python dict

### Research engine
- 10 DuckDuckGo queries per job (skills, tools, MCP, APIs, workflows)
- Up to 4 pages fetched per query, 5000 chars extracted per page
- 14,000 char context window fed to GPT-4o-mini for extraction
- Automatic refinement pass if confidence < 50%
- 7-day cache to avoid redundant searches
""")

st.divider()
st.caption("👈 Open **Config Generator** in the sidebar to generate a new config.")
