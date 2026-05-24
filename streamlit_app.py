"""Streamlit UI for the Product Review Intelligence multi-agent system.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import cfg
from src.crew import build_crew, build_revision_crew
from src.utils.logging_config import configure_logging


st.set_page_config(
    page_title="Product Review Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
  --ink: #15202b;
  --ink-2: #1f2d3a;
  --navy: #0e2a47;
  --navy-2: #1e3a5f;
  --muted: #5a6b7a;
  --muted-2: #8a98a5;
  --line: #e6e0d3;
  --line-2: #cfc6b3;
  --paper: #fbf9f4;
  --paper-2: #f3eee4;
  --card: #ffffff;
  --ochre: #b8860b;
  --ochre-soft: #d9a93b;
  --teal: #0f766e;
  --teal-soft: #2c9c93;
  --brick: #9f1239;
  --brick-soft: #be3455;
  --ok-bg: #ecfccb; --ok-fg: #3f6212; --ok-bd: #bef264;
  --warn-bg: #fef3c7; --warn-fg: #854d0e; --warn-bd: #fcd34d;
  --err-bg: #ffe4e6; --err-fg: #9f1239; --err-bd: #fda4af;
}

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  color: var(--ink);
}

[data-testid="stAppViewContainer"] {
  background: var(--paper);
}

/* hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1180px; }

/* masthead */
.masthead {
  border-top: 4px solid var(--navy);
  border-bottom: 1px solid var(--line);
  padding: 1.4rem 0 1.4rem;
  margin-bottom: 2rem;
  position: relative;
}
.masthead::before {
  content: "";
  position: absolute;
  top: -4px; left: 0;
  width: 140px; height: 4px;
  background: var(--ochre);
}
.masthead .eyebrow {
  font-family: 'Inter', sans-serif;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ochre);
  margin-bottom: 0.55rem;
}
.masthead h1 {
  font-family: 'Source Serif 4', Georgia, serif;
  font-weight: 700;
  font-size: 2.6rem;
  line-height: 1.08;
  letter-spacing: -0.018em;
  color: var(--navy);
  margin: 0 0 0.55rem;
}
.masthead .dek {
  font-family: 'Source Serif 4', Georgia, serif;
  font-style: italic;
  font-size: 1.05rem;
  color: var(--ink-2);
  max-width: 720px;
  margin: 0;
}
.masthead .meta {
  margin-top: 1rem;
  display: flex;
  flex-wrap: wrap;
  gap: 1.4rem;
  font-size: 0.78rem;
  color: var(--muted);
  letter-spacing: 0.02em;
}
.masthead .meta span strong {
  color: var(--navy);
  font-weight: 700;
  margin-left: 0.35rem;
}

/* section headings */
.section {
  font-family: 'Inter', sans-serif;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--navy);
  border-bottom: 2px solid var(--navy);
  padding-bottom: 0.4rem;
  margin: 1.8rem 0 1rem;
  display: flex;
  align-items: center;
  gap: 0.55rem;
}
.section::before {
  content: "";
  display: inline-block;
  width: 8px; height: 8px;
  background: var(--ochre);
  border-radius: 1px;
}

/* roles table */
.roles {
  border: 1px solid var(--line);
  border-radius: 4px;
  overflow: hidden;
  background: var(--card);
}
.roles .row {
  display: grid;
  grid-template-columns: 44px 1fr;
  border-bottom: 1px solid var(--line);
  padding: 1rem 1.1rem;
  background: var(--card);
  transition: background 0.12s;
}
.roles .row:hover { background: var(--paper); }
.roles .row:last-child { border-bottom: none; }
.roles .row.r1 { border-left: 3px solid var(--teal); }
.roles .row.r2 { border-left: 3px solid var(--ochre); }
.roles .row.r3 { border-left: 3px solid var(--brick); }
.roles .num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  font-weight: 600;
  padding-top: 0.1rem;
}
.roles .r1 .num { color: var(--teal); }
.roles .r2 .num { color: var(--ochre); }
.roles .r3 .num { color: var(--brick); }
.roles .role-name {
  font-family: 'Source Serif 4', Georgia, serif;
  font-weight: 700;
  font-size: 1.02rem;
  color: var(--navy);
  margin-bottom: 0.15rem;
}
.roles .role-tools {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  margin-bottom: 0.4rem;
  letter-spacing: 0.01em;
  font-weight: 500;
}
.roles .r1 .role-tools { color: var(--teal); }
.roles .r2 .role-tools { color: var(--ochre); }
.roles .r3 .role-tools { color: var(--brick); }
.roles .role-desc {
  font-size: 0.88rem;
  color: var(--muted);
  line-height: 1.55;
}

/* metrics */
.metric {
  border: 1px solid var(--line);
  border-top: 3px solid var(--teal);
  padding: 0.95rem 1.05rem;
  border-radius: 2px;
  background: var(--card);
}
.metric.m2 { border-top-color: var(--ochre); }
.metric .label {
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
}
.metric .value {
  font-family: 'Source Serif 4', Georgia, serif;
  font-weight: 700;
  font-size: 1.95rem;
  color: var(--navy);
  margin-top: 0.25rem;
  line-height: 1;
}
.metric .sub {
  font-size: 0.72rem;
  color: var(--muted-2);
  margin-top: 0.35rem;
  font-family: 'JetBrains Mono', monospace;
}

/* status pills */
.pill {
  display: inline-flex;
  align-items: center;
  padding: 0.22rem 0.6rem;
  border-radius: 3px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 500;
  border: 1px solid;
  margin-right: 0.35rem;
  margin-bottom: 0.3rem;
}
.pill.ok   { background: var(--ok-bg);   color: var(--ok-fg);   border-color: var(--ok-bd); }
.pill.warn { background: var(--warn-bg); color: var(--warn-fg); border-color: var(--warn-bd); }
.pill.err  { background: var(--err-bg);  color: var(--err-fg);  border-color: var(--err-bd); }

/* sidebar */
section[data-testid="stSidebar"] {
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] > div {
  background: var(--paper-2);
  padding-top: 1rem;
}
section[data-testid="stSidebar"] h3 {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--navy);
  margin-bottom: 0.7rem;
  border-bottom: 2px solid var(--ochre);
  padding-bottom: 0.3rem;
  width: fit-content;
}

/* buttons */
.stButton > button {
  border-radius: 3px;
  font-weight: 600;
  font-size: 0.875rem;
  letter-spacing: 0.01em;
  border: 1px solid var(--line-2);
  background: var(--card);
  color: var(--navy);
  padding: 0.55rem 1rem;
  transition: all 0.12s;
}
.stButton > button:hover {
  border-color: var(--navy);
  background: var(--paper);
  color: var(--navy);
}
button[kind="primary"] {
  background: var(--navy) !important;
  color: #fbf9f4 !important;
  border: 1px solid var(--navy) !important;
  box-shadow: 2px 2px 0 var(--ochre);
}
button[kind="primary"]:hover {
  background: var(--navy-2) !important;
  border-color: var(--navy-2) !important;
  transform: translate(-1px, -1px);
  box-shadow: 3px 3px 0 var(--ochre);
}
button[kind="primary"]:disabled {
  background: var(--muted-2) !important;
  border-color: var(--muted-2) !important;
  box-shadow: none !important;
  color: var(--paper) !important;
}
.stDownloadButton > button {
  border-radius: 3px;
  font-size: 0.825rem;
  font-weight: 600;
  color: var(--navy);
}

/* dataframe */
[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 4px;
}

/* expander */
[data-testid="stExpander"] {
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--paper);
}
[data-testid="stExpander"] summary {
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  color: var(--ink-2);
}

/* brief display */
.brief {
  border: 1px solid var(--line);
  border-top: 4px solid var(--navy);
  background: var(--card);
  padding: 2rem 2.4rem;
  border-radius: 2px;
  font-family: 'Source Serif 4', Georgia, serif;
  line-height: 1.65;
  color: var(--ink);
  box-shadow: 4px 4px 0 var(--paper-2);
  position: relative;
}
.brief::before {
  content: "";
  position: absolute;
  top: -4px; left: 0;
  width: 120px; height: 4px;
  background: var(--ochre);
}
.brief h1 {
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 1.7rem;
  font-weight: 700;
  color: var(--navy);
  margin: 0 0 1.2rem;
  border-bottom: 1px solid var(--line);
  padding-bottom: 0.6rem;
}
.brief h2 {
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 1.08rem;
  font-weight: 700;
  color: var(--navy);
  margin-top: 1.4rem;
  margin-bottom: 0.5rem;
}
.brief h3 {
  font-family: 'Inter', sans-serif;
  font-size: 0.74rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ochre);
  margin-top: 1rem;
  margin-bottom: 0.4rem;
  font-weight: 700;
}
.brief a { color: var(--teal); }

/* radio styling */
div[role="radiogroup"] label {
  font-size: 0.875rem;
}

/* HITL banner */
.hitl-note {
  background: #fff8e6;
  border-left: 4px solid var(--ochre);
  padding: 0.85rem 1.1rem;
  font-size: 0.88rem;
  color: var(--ink-2);
  border-radius: 0 3px 3px 0;
  margin-bottom: 0.9rem;
}
.hitl-note strong { color: var(--ochre); }

/* colophon */
.colophon {
  margin-top: 3.5rem;
  padding-top: 1.4rem;
  border-top: 3px double var(--line-2);
  font-size: 0.74rem;
  color: var(--muted);
  letter-spacing: 0.04em;
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.6rem;
}
.colophon strong { color: var(--navy); font-weight: 700; }

/* alerts */
div[data-testid="stAlert"] {
  border-radius: 4px;
  border-left-width: 3px;
}

/* caption */
.stCaption, [data-testid="stCaptionContainer"] {
  font-size: 0.78rem;
  color: var(--muted);
}
</style>
""",
    unsafe_allow_html=True,
)


def _init_state() -> None:
    defaults = {
        "stage": "idle",
        "draft": "",
        "final_brief": "",
        "log_path": None,
        "reviews_path": None,
        "product_name": "",
        "elapsed": 0.0,
        # cached specialist outputs (for fast revisions)
        "analyst_output": "",
        "researcher_output": "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _extract_task_output(crew_obj, idx: int) -> str:
    """Pull the raw text output of task #idx from a crew that has just run."""
    try:
        task = crew_obj.tasks[idx]
        out = getattr(task, "output", None)
        if out is None:
            return ""
        return getattr(out, "raw", "") or str(out)
    except (IndexError, AttributeError):
        return ""


_init_state()


def _llm_summary() -> tuple[str, str]:
    """Return (provider, model) tuple from cfg."""
    m = cfg.model
    if "/" in m:
        provider, model = m.split("/", 1)
        return provider.capitalize(), model
    return "—", m


def _llm_ready() -> bool:
    """True if the configured LLM backend has what it needs to run."""
    if cfg.model.startswith("gemini/"):
        return bool(cfg.gemini_api_key)
    return True  # Ollama: assume reachable; CrewAI will surface errors


provider, model_name = _llm_summary()
bert_present = (cfg.model_dir / "config.json").exists()


# ──────────────────────── masthead ────────────────────────
st.markdown(
    f"""
<div class="masthead">
  <div class="eyebrow">Université Internationale de Rabat · S8 Integrated Project</div>
  <h1>Product Review Intelligence</h1>
  <p class="dek">A multi-agent system for surfacing customer sentiment, competitive context, and an editorial brief — backed by a fine-tuned DistilBERT classifier.</p>
  <div class="meta">
    <span>Framework<strong>CrewAI</strong></span>
    <span>Classifier<strong>DistilBERT (fine-tuned)</strong></span>
    <span>LLM backend<strong>{provider} · {model_name}</strong></span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ──────────────────────── sidebar ────────────────────────
with st.sidebar:
    st.markdown("### Configuration")

    product = st.text_input("Product under review", value="Wireless Earbuds X")

    st.markdown("### Reviews source")
    source = st.radio(
        "Source",
        ["Bundled sample (10 reviews)", "Upload CSV", "Paste manually"],
        index=0,
        label_visibility="collapsed",
    )

    uploaded_path: Path | None = None
    if source.startswith("Bundled"):
        uploaded_path = cfg.data_dir / "processed" / "sample_reviews.csv"
        if not uploaded_path.exists():
            st.warning("Sample CSV missing — run `python scripts/make_sample_reviews.py`.")
    elif source == "Upload CSV":
        f = st.file_uploader("CSV with a `text` column", type=["csv"], label_visibility="collapsed")
        if f is not None:
            uploaded_path = cfg.data_dir / "processed" / f"upload_{int(time.time())}.csv"
            uploaded_path.write_bytes(f.read())
            st.caption(f"Saved → `{uploaded_path.name}`")
    else:
        text = st.text_area(
            "One review per line",
            height=180,
            placeholder="Battery dies fast.\nGreat sound.\n…",
        )
        if text.strip():
            uploaded_path = cfg.data_dir / "processed" / f"pasted_{int(time.time())}.csv"
            with uploaded_path.open("w", newline="", encoding="utf-8") as out:
                w = csv.writer(out)
                w.writerow(["id", "text"])
                for i, line in enumerate(l for l in text.splitlines() if l.strip()):
                    w.writerow([i + 1, line.strip()])

    st.markdown("### System")

    if _llm_ready():
        st.markdown(f'<span class="pill ok">LLM ready · {provider}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill err">LLM not configured</span>', unsafe_allow_html=True)
        st.caption("Set `GEMINI_API_KEY` in `.env` or run a local Ollama server.")

    if bert_present:
        st.markdown('<span class="pill ok">BERT loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill warn">BERT missing · heuristic fallback</span>', unsafe_allow_html=True)


# ──────────────────────── main grid ────────────────────────
left, right = st.columns([1.2, 1], gap="large")

with left:
    st.markdown('<div class="section">Reviews dataset</div>', unsafe_allow_html=True)
    if uploaded_path and uploaded_path.exists():
        try:
            df = pd.read_csv(uploaded_path)
            st.dataframe(df.head(20), height=320, use_container_width=True, hide_index=True)
            st.caption(f"{len(df)} review(s) · source: {uploaded_path.name}")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    else:
        st.info("Select a reviews source in the sidebar.")

    can_run = (
        uploaded_path is not None
        and uploaded_path.exists()
        and product.strip()
        and _llm_ready()
        and st.session_state.stage in ("idle", "done")
    )

    run_clicked = st.button(
        "Run multi-agent analysis",
        type="primary",
        disabled=not can_run,
        use_container_width=True,
    )

    if run_clicked and can_run:
        st.session_state.stage = "drafting"
        st.session_state.product_name = product
        st.session_state.reviews_path = str(uploaded_path)

        run_id = time.strftime("%Y%m%d_%H%M%S")
        st.session_state.log_path = str(configure_logging(run_id))

        t0 = time.time()
        with st.status("Pipeline running…", expanded=True) as status:
            st.write("Step 1/3 · Sentiment Analyst — loading reviews, calling BERT")
            st.write("Step 2/3 · Market Researcher — querying the public web")
            st.write("Step 3/3 · Orchestrator — drafting the editorial brief")
            try:
                crew = build_crew(product, str(uploaded_path), with_hitl=False)
                result = crew.kickoff()
                st.session_state.draft = str(result)
                st.session_state.analyst_output = _extract_task_output(crew, 0)
                st.session_state.researcher_output = _extract_task_output(crew, 1)
                st.session_state.elapsed = time.time() - t0
                st.session_state.stage = "review"
                status.update(
                    label=f"Draft ready in {st.session_state.elapsed:.1f}s — awaiting your approval",
                    state="complete",
                )
            except Exception as e:
                st.session_state.stage = "idle"
                status.update(label=f"Run failed: {e}", state="error")
                st.exception(e)

with right:
    st.markdown('<div class="section">Agent roster</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="roles">
  <div class="row r1">
    <div class="num">01</div>
    <div>
      <div class="role-name">Sentiment Analyst</div>
      <div class="role-tools">tools · bert_sentiment, review_loader</div>
      <div class="role-desc">Owns the fine-tuned DistilBERT classifier. Calls the model once per review, aggregates labels, and surfaces the dominant complaints and praises.</div>
    </div>
  </div>
  <div class="row r2">
    <div class="num">02</div>
    <div>
      <div class="role-name">Market Researcher</div>
      <div class="role-tools">tools · competitor_search</div>
      <div class="role-desc">Uses the analyst's complaint themes as search seeds. Returns three plausible competitors with evidence URLs — no fabrication permitted.</div>
    </div>
  </div>
  <div class="row r3">
    <div class="num">03</div>
    <div>
      <div class="role-name">Report Orchestrator</div>
      <div class="role-tools">tools · (none — synthesis only)</div>
      <div class="role-desc">Compiles both specialists' outputs into a one-page market brief, then yields to a human-in-the-loop checkpoint before finalisation.</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if bert_present:
        try:
            metrics = json.loads((cfg.output_dir / "eval_metrics.json").read_text())
            st.markdown('<div class="section" style="margin-top:1.5rem">Model performance</div>', unsafe_allow_html=True)
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown(
                    f'<div class="metric"><div class="label">Accuracy</div>'
                    f'<div class="value">{metrics["accuracy"] * 100:.1f}<span style="font-size:1.1rem;color:var(--muted-2)">%</span></div>'
                    f'<div class="sub">n = {metrics.get("n_test", "—")} held-out</div></div>',
                    unsafe_allow_html=True,
                )
            with mc2:
                st.markdown(
                    f'<div class="metric m2"><div class="label">Weighted F1</div>'
                    f'<div class="value">{metrics["f1_weighted"]:.3f}</div>'
                    f'<div class="sub">balanced classes</div></div>',
                    unsafe_allow_html=True,
                )
        except FileNotFoundError:
            pass


# ──────────────────────── HITL ────────────────────────
if st.session_state.stage == "review":
    st.markdown('<div class="section">Human-in-the-loop checkpoint</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hitl-note">'
        "<strong>Checkpoint.</strong> The orchestrator has produced a draft. "
        "Approve it, edit the text directly, or send written instructions for a revision."
        "</div>",
        unsafe_allow_html=True,
    )

    edited = st.text_area(
        "Draft brief",
        value=st.session_state.draft,
        height=380,
        key="draft_editor",
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns(2)
    if col_a.button("Approve as-is", type="primary", use_container_width=True):
        st.session_state.final_brief = st.session_state.draft
        st.session_state.stage = "done"
        st.rerun()
    if col_b.button("Save my edits", use_container_width=True):
        st.session_state.final_brief = edited
        st.session_state.stage = "done"
        st.rerun()

    st.markdown(
        '<div style="margin-top:1.4rem; font-family:Inter,sans-serif; '
        'font-size:0.72rem; font-weight:700; letter-spacing:0.14em; '
        'text-transform:uppercase; color:var(--navy); '
        'border-bottom:1px solid var(--line); padding-bottom:0.3rem; '
        'margin-bottom:0.8rem;">'
        'Or — ask the orchestrator to revise'
        '</div>',
        unsafe_allow_html=True,
    )

    feedback = st.text_input(
        "Instructions to the orchestrator",
        placeholder="e.g. make it shorter · emphasise the battery issue · add a risk bullet",
        key="feedback_input",
        label_visibility="collapsed",
    )

    col_c, col_d = st.columns(2)
    if col_c.button(
        "Revise with these instructions",
        use_container_width=True,
        disabled=not feedback.strip(),
    ):
        st.session_state.stage = "drafting"
        t0 = time.time()
        with st.status("Orchestrator revising the brief…", expanded=True) as s:
            s.write(f"Reviewer instruction: « {feedback.strip()} »")
            s.write("Re-using cached Sentiment Analyst + Market Researcher outputs")
            s.write("Only the Orchestrator is re-invoked — much faster")
            crew = build_revision_crew(
                product_name=st.session_state.product_name,
                analyst_output=st.session_state.analyst_output,
                researcher_output=st.session_state.researcher_output,
                current_draft=st.session_state.draft,
                feedback=feedback.strip(),
            )
            st.session_state.draft = str(crew.kickoff())
            s.update(label=f"Revised in {time.time()-t0:.1f}s", state="complete")
        st.session_state.stage = "review"
        st.rerun()
    if col_d.button("Regenerate from scratch", use_container_width=True):
        st.session_state.stage = "drafting"
        with st.status("Re-running full pipeline…", expanded=True):
            crew = build_crew(
                st.session_state.product_name,
                st.session_state.reviews_path,
                with_hitl=False,
            )
            st.session_state.draft = str(crew.kickoff())
            st.session_state.analyst_output = _extract_task_output(crew, 0)
            st.session_state.researcher_output = _extract_task_output(crew, 1)
        st.session_state.stage = "review"
        st.rerun()


# ──────────────────────── final brief ────────────────────────
if st.session_state.stage == "done" and st.session_state.final_brief:
    st.markdown('<div class="section">Market brief</div>', unsafe_allow_html=True)

    out_path = cfg.output_dir / f"market_brief_{time.strftime('%Y%m%d_%H%M%S')}.md"
    if not out_path.exists():
        out_path.write_text(st.session_state.final_brief, encoding="utf-8")

    st.markdown(f'<div class="brief">\n\n{st.session_state.final_brief}\n\n</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1.2rem"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    c1.download_button(
        "Download brief (.md)",
        data=st.session_state.final_brief,
        file_name=out_path.name,
        mime="text/markdown",
        use_container_width=True,
    )
    if st.session_state.log_path and Path(st.session_state.log_path).exists():
        c2.download_button(
            "Download action log (.jsonl)",
            data=Path(st.session_state.log_path).read_bytes(),
            file_name=Path(st.session_state.log_path).name,
            mime="application/x-ndjson",
            use_container_width=True,
        )
    if c3.button("Start new run", use_container_width=True):
        for k in ("stage", "draft", "final_brief", "log_path", "analyst_output", "researcher_output"):
            st.session_state[k] = "idle" if k == "stage" else ("" if isinstance(st.session_state[k], str) else None)
        st.rerun()


# ──────────────────────── log tail ────────────────────────
if st.session_state.log_path and Path(st.session_state.log_path).exists():
    with st.expander("Agent action log · JSONL", expanded=False):
        lines = Path(st.session_state.log_path).read_text(encoding="utf-8").strip().splitlines()
        st.caption(f"{len(lines)} events · {Path(st.session_state.log_path).name}")
        st.code("\n".join(lines[-30:]) or "(no events yet)", language="json")


# ──────────────────────── colophon ────────────────────────
st.markdown(
    '<div class="colophon">'
    '<span><strong>UIR · S8 Integrated Project</strong> — Building Multi-Agent AI Systems</span>'
    '<span>Prof. Hakim Hafidi · AI &amp; Big Data · 2025–2026</span>'
    '</div>',
    unsafe_allow_html=True,
)
