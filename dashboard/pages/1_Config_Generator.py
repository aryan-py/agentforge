"""Config Generator — research a job description and produce downloadable agent config files."""

import asyncio
import os
import sys
from pathlib import Path

import nest_asyncio
import streamlit as st

asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
nest_asyncio.apply()
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Load .env into os.environ before any client is instantiated
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from agentforge.config_generator.generator import FRAMEWORK_LABELS, Framework  # noqa: E402
from agentforge.config_generator.mcp_catalog import find_servers_for_tool_types  # noqa: E402
from agentforge.config_generator.packager import _CONFIG_FILENAME  # noqa: E402

st.set_page_config(
    page_title="Config Generator — AgentForge",
    page_icon="🧩",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.step-done   { color: #34c78a; font-weight: 600; }
.step-active { color: #4f8ef7; font-weight: 700; font-size: 1.05em; }
.step-todo   { color: #888; }
.skill-chip  {
    display: inline-block; background: #1e3a5f; color: #7eb8f7;
    border-radius: 12px; padding: 2px 10px; margin: 2px; font-size: 0.82em;
}
.tool-chip  {
    display: inline-block; background: #1e3a2a; color: #5ecf8a;
    border-radius: 12px; padding: 2px 10px; margin: 2px; font-size: 0.82em;
}
.source-chip {
    display: inline-block; background: #2a2a2a; color: #aaa;
    border-radius: 6px; padding: 1px 8px; margin: 1px; font-size: 0.75em;
}
</style>
""", unsafe_allow_html=True)

st.title("🧩 Agent Config Generator")
st.caption(
    "Describe your agent's job. AgentForge searches the web across 10 targeted queries "
    "to discover the exact skills, tools, and MCP servers it needs — then generates "
    "ready-to-use config files you can drop straight into your project."
)
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "cg_step": 1,
    "cg_research": None,
    "cg_servers": None,
    "cg_all_servers": None,
    "cg_framework": None,
    "cg_role": "",
    "cg_extra_reqs": "",
    "cg_zip": None,
    "cg_description": "",
    "cg_tenant": "demo",
    "cg_sources_count": 0,
    "cg_queries_count": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def reset():
    for k in list(st.session_state.keys()):
        if k.startswith("cg_"):
            del st.session_state[k]
    st.rerun()


# ── Progress bar ──────────────────────────────────────────────────────────────
STEPS = ["Describe", "Research", "Configure", "Review & Download"]
step = st.session_state.cg_step
cols = st.columns(len(STEPS))
for i, (col, label) in enumerate(zip(cols, STEPS), 1):
    if i < step:
        col.markdown(f"<div class='step-done' style='text-align:center'>✅ {label}</div>", unsafe_allow_html=True)
    elif i == step:
        col.markdown(f"<div class='step-active' style='text-align:center'>▶ {label}</div>", unsafe_allow_html=True)
    else:
        col.markdown(f"<div class='step-todo' style='text-align:center'>{i}. {label}</div>", unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Input
# ══════════════════════════════════════════════════════════════════════════════
if step == 1:
    st.subheader("Step 1 — Describe your agent's job")

    col_form, col_tips = st.columns([2, 1])

    with col_form:
        with st.form("cg_input_form"):
            description = st.text_area(
                "Job description",
                value=st.session_state.cg_description,
                placeholder=(
                    "e.g. Monitor a GitHub repository for new issues, search the web for known "
                    "solutions and similar bugs, then post a Slack digest every morning with "
                    "recommended fixes and priority ranking."
                ),
                height=160,
            )
            tenant_id = st.text_input("Tenant / project name (optional)", value=st.session_state.cg_tenant)
            go = st.form_submit_button("🔍 Research this job", type="primary", use_container_width=True)

        if go:
            if not description.strip():
                st.warning("Please enter a job description.")
            else:
                st.session_state.cg_description = description.strip()
                st.session_state.cg_tenant = tenant_id.strip() or "demo"
                st.session_state.cg_step = 2
                st.rerun()

    with col_tips:
        st.markdown("**💡 Tips for better results**")
        st.markdown("""
- Mention the **domain** (finance, DevOps, healthcare…)
- Mention specific **integrations** (GitHub, Slack, Postgres…)
- Mention the **output** (report, alert, summary, dashboard…)
- The more specific you are, the better the config
        """)
        st.info("Research runs **10 queries** across skills, tools, MCP servers, APIs, and best practices.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Research
# ══════════════════════════════════════════════════════════════════════════════
elif step == 2:
    st.subheader("Step 2 — Deep web research in progress…")
    desc_preview = st.session_state.cg_description
    st.info(f"**Job:** {desc_preview[:160]}{'…' if len(desc_preview) > 160 else ''}")

    progress = st.progress(0, text="Initializing research engine…")
    status_area = st.empty()

    try:
        from agentforge.config.settings import settings
        from agentforge.research.cache import ResearchCache
        from agentforge.research.result_parser import ResearchResultParser
        from agentforge.research.web_searcher import WebSearcher
        from agentforge.schemas.job import JobDefinition
        from langchain_openai import ChatOpenAI

        fast_llm = ChatOpenAI(model=settings.ROUTER_MODEL, temperature=0)
        searcher = WebSearcher(max_results_per_query=6, fetch_pages=True)
        parser = ResearchResultParser(llm=fast_llm)
        cache = ResearchCache(ttl_days=settings.RESEARCH_CACHE_TTL_DAYS)

        progress.progress(10, text="Classifying job type and domain…")
        from agentforge.core.meta_agent import MetaAgent
        meta = MetaAgent(llm=fast_llm)
        job = asyncio.run(
            meta.classify(st.session_state.cg_description, st.session_state.cg_tenant)
        )
        status_area.markdown(f"📌 **Job type:** `{job.job_type}` · **Domain:** `{job.domain}`")

        progress.progress(20, text="Checking research cache…")
        cached = asyncio.run(cache.get(job.job_type, job.domain))
        if cached and cached.confidence >= 0.6:
            research = cached
            progress.progress(100, text="✅ Loaded from cache")
            status_area.success(f"✅ Used cached research ({len(research.sources)} sources · confidence {cached.confidence:.0%})")
        else:
            progress.progress(30, text="Running 10 targeted web searches…")
            status_area.markdown("🔍 Searching across skills, tools, MCP servers, APIs, and workflows…")
            search_results = asyncio.run(
                searcher.research_job(job.job_type, job.domain, job.description)
            )

            progress.progress(65, text=f"Analyzing {len(search_results)} sources with AI…")
            status_area.markdown(f"🧠 Analyzing **{len(search_results)} unique sources** — extracting skills, tools, and packages…")
            research = asyncio.run(parser.parse(search_results, job))

            if research.confidence < 0.5:
                progress.progress(80, text="Confidence low — running refinement pass…")
                status_area.markdown("🔄 Low confidence — running targeted refinement search…")
                research = asyncio.run(
                    parser.refine(research, f"{job.domain} {job.job_type} tools packages best practices")
                )

            progress.progress(90, text="Caching results…")
            asyncio.run(cache.set(job.job_type, job.domain, research))

        progress.progress(95, text="Matching MCP servers from catalog…")
        servers = find_servers_for_tool_types(research.required_tool_types)

        st.session_state.cg_research = research
        st.session_state.cg_servers = servers
        st.session_state.cg_all_servers = servers
        st.session_state.cg_sources_count = len(research.sources)
        st.session_state.cg_queries_count = 10
        st.session_state.cg_step = 3

        progress.progress(100, text="Research complete!")
        st.rerun()

    except Exception as e:
        st.error(f"Research failed: {e}")
        st.exception(e)
        if st.button("← Try again"):
            st.session_state.cg_step = 1
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Configure
# ══════════════════════════════════════════════════════════════════════════════
elif step == 3:
    research = st.session_state.cg_research
    servers = st.session_state.cg_all_servers

    st.subheader("Step 3 — Review research & configure your agent")

    # ── Research summary ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Research confidence", f"{research.confidence:.0%}")
    m2.metric("Skills discovered", len(research.required_skills))
    m3.metric("Tool types needed", len(research.required_tool_types))
    m4.metric("MCP servers matched", len(servers))

    with st.expander("📊 Full research findings", expanded=True):
        tab_skills, tab_tools, tab_approach, tab_sources = st.tabs(
            ["🎯 Skills", "🔧 Tools & Types", "📋 Expert Approach", "🔗 Sources"]
        )

        with tab_skills:
            st.markdown("**Skills the research found for this job:**")
            chips = " ".join(
                f"<span class='skill-chip'>{s}</span>"
                for s in research.required_skills
            )
            st.markdown(chips or "_No skills found_", unsafe_allow_html=True)
            if research.domain_knowledge_summary:
                st.divider()
                st.markdown("**Domain knowledge summary:**")
                st.markdown(research.domain_knowledge_summary)

        with tab_tools:
            st.markdown("**Tool types required:**")
            chips = " ".join(
                f"<span class='tool-chip'>{t}</span>"
                for t in research.required_tool_types
            )
            st.markdown(chips or "_No tool types found_", unsafe_allow_html=True)

            if research.suggested_mcp_servers:
                st.divider()
                st.markdown("**Packages & libraries discovered by research:**")
                for pkg in research.suggested_mcp_servers:
                    st.markdown(f"  • `{pkg}`")

        with tab_approach:
            st.markdown("**How an expert would complete this job:**")
            for i, s in enumerate(research.expert_approach, 1):
                st.markdown(f"{i}. {s}")

        with tab_sources:
            st.markdown(f"**{len(research.sources)} sources consulted:**")
            for url in research.sources[:30]:
                st.markdown(f"<span class='source-chip'>{url[:80]}</span>", unsafe_allow_html=True)

    st.divider()

    # ── Questions form ────────────────────────────────────────────────────────
    with st.form("cg_questions_form"):
        st.markdown("### ⚙️ Configure your agent")

        col_fw, col_role = st.columns(2)
        with col_fw:
            st.markdown("**Target framework**")
            framework_choice = st.selectbox(
                "Framework",
                options=list(FRAMEWORK_LABELS.keys()),
                format_func=lambda k: FRAMEWORK_LABELS[k],
                label_visibility="collapsed",
            )
        with col_role:
            st.markdown("**Custom agent role title** *(optional)*")
            role_input = st.text_input(
                "Role title",
                placeholder=f"e.g. Senior {research.domain.title()} Specialist",
                label_visibility="collapsed",
            )

        st.divider()
        st.markdown("**Additional constraints or requirements** *(optional)*")
        st.caption("These will be appended to the constraints section of your system prompt.")
        extra_reqs = st.text_area(
            "Additional requirements",
            placeholder="e.g. Always respond in English. Never access URLs outside the approved list.",
            height=80,
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Select MCP servers to include**")
        st.caption(
            "🟢 **FREE** = works immediately, no API key needed. &nbsp;&nbsp;"
            "🔑 **NEEDS KEY** = requires an API key (instructions included in output).",
            unsafe_allow_html=True,
        )

        selected_ids = []
        if servers:
            free = [s for s in servers if not s.requires_key]
            keyed = [s for s in servers if s.requires_key]

            if free:
                st.markdown("**✅ Free — no setup needed:**")
                for s in free:
                    if st.checkbox(f"**{s.name}** — {s.description}", value=True, key=f"srv_{s.id}"):
                        selected_ids.append(s.id)
                    if s.extra_placeholders:
                        for note in s.extra_placeholders.values():
                            st.caption(f"  ⚠️  {note}")

            if keyed:
                st.markdown("**🔑 Requires API key:**")
                for s in keyed:
                    if st.checkbox(
                        f"**{s.name}** (`{s.key_name}`) — {s.description}",
                        value=True,
                        key=f"srv_{s.id}",
                    ):
                        selected_ids.append(s.id)
                    if s.key_url:
                        st.caption(f"  Get key: [{s.key_url}]({s.key_url})")
                    elif s.key_instructions:
                        st.caption(f"  {s.key_instructions}")
        else:
            st.info("No MCP servers matched from the catalog. You can add custom ones after downloading.")

        col_submit, col_back = st.columns([4, 1])
        with col_submit:
            submitted = st.form_submit_button("⚙️ Generate config files", type="primary", use_container_width=True)
        with col_back:
            back = st.form_submit_button("← Back", use_container_width=True)

    if back:
        st.session_state.cg_step = 1
        st.rerun()

    if submitted:
        st.session_state.cg_framework = framework_choice
        st.session_state.cg_role = role_input.strip()
        st.session_state.cg_extra_reqs = extra_reqs.strip()
        st.session_state.cg_servers = [s for s in servers if s.id in selected_ids]
        st.session_state.cg_step = 4
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Review & Download
# ══════════════════════════════════════════════════════════════════════════════
elif step == 4:
    import io
    import zipfile

    from agentforge.config_generator.generator import (
        generate_mcp_config,
        generate_readme,
        generate_requirements_txt,
        generate_skills_yaml,
        generate_system_prompt,
    )

    research = st.session_state.cg_research
    servers = st.session_state.cg_servers
    framework = st.session_state.cg_framework
    role = st.session_state.cg_role
    extra_reqs = st.session_state.cg_extra_reqs

    st.subheader("Step 4 — Review & Download")

    # Stats bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Framework", FRAMEWORK_LABELS[framework])
    m2.metric("MCP servers", len(servers))
    m3.metric("Skills", len(research.required_skills))
    m4.metric("Research confidence", f"{research.confidence:.0%}")

    # ── Generate files ────────────────────────────────────────────────────────
    system_prompt_text = generate_system_prompt(research, role)
    if extra_reqs:
        system_prompt_text = system_prompt_text.replace(
            "- [ADD YOUR OWN CONSTRAINTS HERE]",
            "\n".join(f"- {line.strip()}" for line in extra_reqs.splitlines() if line.strip()),
        )

    mcp_text = generate_mcp_config(servers, framework)
    skills_text = generate_skills_yaml(research)
    reqs_text = generate_requirements_txt(servers, research)
    readme_text = generate_readme(research, servers, framework)

    # ── File preview tabs ─────────────────────────────────────────────────────
    st.markdown("### 📁 Generated files preview")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 system_prompt.txt",
        "🔌 mcp_config",
        "📋 skills.yaml",
        "📦 requirements.txt",
        "📖 README.md",
    ])
    with tab1:
        st.code(system_prompt_text, language="text")
    with tab2:
        lang = "json" if framework in ("claude_desktop", "cursor", "windsurf") else "python"
        st.code(mcp_text, language=lang)
    with tab3:
        st.code(skills_text, language="yaml")
    with tab4:
        st.code(reqs_text, language="text")
    with tab5:
        st.markdown(readme_text)

    # ── MCP server summary ────────────────────────────────────────────────────
    if servers:
        st.divider()
        st.markdown("### 🔌 MCP Servers included")
        free = [s for s in servers if not s.requires_key]
        keyed = [s for s in servers if s.requires_key]

        col_free, col_keyed = st.columns(2)
        with col_free:
            if free:
                st.markdown("**✅ Ready to use:**")
                for s in free:
                    st.markdown(f"- **{s.name}** — {s.description}")
                    if s.extra_placeholders:
                        for note in s.extra_placeholders.values():
                            st.caption(f"  ⚠️  {note}")
        with col_keyed:
            if keyed:
                st.markdown("**🔑 Requires API key:**")
                for s in keyed:
                    line = f"- **{s.name}** (`{s.key_name}`)"
                    if s.key_url:
                        line += f" — [Get key ↗]({s.key_url})"
                    st.markdown(line)
                    if s.key_instructions:
                        st.caption(f"  {s.key_instructions}")

    # ── Download ──────────────────────────────────────────────────────────────
    st.divider()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_CONFIG_FILENAME[framework], mcp_text)
        zf.writestr("system_prompt.txt", system_prompt_text)
        zf.writestr("skills.yaml", skills_text)
        zf.writestr("requirements.txt", reqs_text)
        zf.writestr("README.md", readme_text)
    zip_bytes = buf.getvalue()

    domain = research.domain.lower().replace(" ", "_")
    job_type = research.job_type.lower().replace(" ", "_")
    zip_name = f"agentforge_{domain}_{job_type}_{framework}.zip"

    # Log to history
    import json as _json
    from datetime import datetime as _dt
    _hist_path = Path(__file__).parent.parent.parent / "logs" / "config_history.jsonl"
    _hist_path.parent.mkdir(parents=True, exist_ok=True)
    if not st.session_state.get("cg_logged"):
        with _hist_path.open("a") as _f:
            _f.write(_json.dumps({
                "generated_at": _dt.utcnow().isoformat(),
                "description": st.session_state.cg_description[:200],
                "domain": research.domain,
                "job_type": research.job_type,
                "framework": framework,
                "skills_count": len(research.required_skills),
                "tools_count": len(research.required_tool_types),
                "mcp_servers_count": len(servers),
                "confidence": round(research.confidence, 2),
                "zip_name": zip_name,
            }) + "\n")
        st.session_state.cg_logged = True

    col_dl, col_back, col_reset = st.columns([3, 1, 1])
    with col_dl:
        st.download_button(
            label="⬇️ Download all files as ZIP",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.cg_step = 3
            st.rerun()
    with col_reset:
        if st.button("🔄 New job", use_container_width=True):
            reset()

    st.success(
        f"**{zip_name}** is ready. Unzip into your project, follow the README, "
        f"and replace any `YOUR_..._HERE` placeholders."
    )
