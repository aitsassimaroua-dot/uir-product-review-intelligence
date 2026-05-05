"""Streamlit UI for the Product Review Intelligence multi-agent system.

Run:
    streamlit run streamlit_app.py

The CLI (python -m src.main) keeps CrewAI's terminal HITL. This UI replaces it
with approve/edit/reject buttons — same checkpoint, friendlier surface.
"""
from __future__ import annotations

import csv
import json
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import cfg
from src.crew import build_crew
from src.utils.logging_config import configure_logging


st.set_page_config(
    page_title="Product Review Intelligence",
    page_icon="🛒",
    layout="wide",
)

# ──────────────────────── session state ────────────────────────
def _init_state():
    defaults = {
        "stage": "idle",          # idle | drafting | review | done
        "draft": "",
        "final_brief": "",
        "log_path": None,
        "reviews_path": None,
        "product_name": "",
        "elapsed": 0.0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

_init_state()


# ──────────────────────── header ────────────────────────
st.title("🛒 Product Review Intelligence")
st.caption(
    "Multi-agent AI system · UIR S8 Integrated Project · "
    "Sentiment Analyst (BERT) + Market Researcher + Orchestrator"
)

# ──────────────────────── sidebar: inputs ────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    product = st.text_input("Product name", value="Wireless Earbuds X")

    st.markdown("---")
    source = st.radio(
        "Reviews source",
        ["Sample CSV (10 reviews)", "Upload CSV", "Paste manually"],
        index=0,
    )

    uploaded_path: Path | None = None
    if source == "Sample CSV (10 reviews)":
        uploaded_path = cfg.data_dir / "processed" / "sample_reviews.csv"
        if not uploaded_path.exists():
            st.warning("Sample CSV missing — run `python scripts/make_sample_reviews.py`.")
    elif source == "Upload CSV":
        f = st.file_uploader("CSV with at least a `text` column", type=["csv"])
        if f is not None:
            uploaded_path = cfg.data_dir / "processed" / f"upload_{int(time.time())}.csv"
            uploaded_path.write_bytes(f.read())
            st.success(f"Saved to {uploaded_path.name}")
    else:
        text = st.text_area("One review per line", height=200, placeholder="Battery dies fast.\nGreat sound.\n...")
        if text.strip():
            uploaded_path = cfg.data_dir / "processed" / f"pasted_{int(time.time())}.csv"
            with uploaded_path.open("w", newline="", encoding="utf-8") as out:
                w = csv.writer(out)
                w.writerow(["id", "text"])
                for i, line in enumerate(l for l in text.splitlines() if l.strip()):
                    w.writerow([i + 1, line.strip()])

    st.markdown("---")
    if cfg.gemini_api_key:
        st.success("✅ Gemini API key configured")
    else:
        st.error("❌ GEMINI_API_KEY missing in `.env`")

    if (cfg.model_dir / "config.json").exists():
        st.success("✅ BERT model loaded")
    else:
        st.warning("⚠️ No BERT checkpoint — tool falls back to heuristic")


# ──────────────────────── main: preview + run ────────────────────────
left, right = st.columns([1, 1])

with left:
    st.subheader("📋 Reviews preview")
    if uploaded_path and uploaded_path.exists():
        try:
            df = pd.read_csv(uploaded_path)
            st.dataframe(df.head(20), height=320, use_container_width=True)
            st.caption(f"{len(df)} review(s) · `{uploaded_path.name}`")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    else:
        st.info("Pick a reviews source in the sidebar.")

    can_run = (
        uploaded_path is not None
        and uploaded_path.exists()
        and product.strip()
        and cfg.gemini_api_key
        and st.session_state.stage in ("idle", "done")
    )

    run_clicked = st.button("🚀 Run multi-agent analysis", type="primary", disabled=not can_run)

    if run_clicked and can_run:
        st.session_state.stage = "drafting"
        st.session_state.product_name = product
        st.session_state.reviews_path = str(uploaded_path)

        run_id = time.strftime("%Y%m%d_%H%M%S")
        st.session_state.log_path = str(configure_logging(run_id))

        t0 = time.time()
        with st.status("Agents working…", expanded=True) as status:
            st.write("📥 Loading reviews + Sentiment Analyst calling BERT…")
            st.write("🔎 Market Researcher querying DuckDuckGo…")
            st.write("✍️ Orchestrator drafting the brief…")
            try:
                crew = build_crew(product, str(uploaded_path), with_hitl=False)
                result = crew.kickoff()
                st.session_state.draft = str(result)
                st.session_state.elapsed = time.time() - t0
                st.session_state.stage = "review"
                status.update(label=f"✅ Draft ready in {st.session_state.elapsed:.1f}s — please review", state="complete")
            except Exception as e:
                st.session_state.stage = "idle"
                status.update(label=f"❌ Run failed: {e}", state="error")
                st.exception(e)

with right:
    st.subheader("🤖 Architecture")
    st.markdown("""
- **Sentiment Analyst** — owns the BERT sentiment tool + CSV loader
- **Market Researcher** — owns the DuckDuckGo search tool
- **Orchestrator** — synthesises both outputs into the brief (no tools)
- **HITL** — you approve / edit / reject the draft below
""")

    if (cfg.model_dir / "config.json").exists():
        try:
            metrics = json.loads((cfg.output_dir / "eval_metrics.json").read_text())
            c1, c2 = st.columns(2)
            c1.metric("BERT accuracy", f"{metrics['accuracy']*100:.1f}%")
            c2.metric("Weighted F1", f"{metrics['f1_weighted']:.3f}")
        except FileNotFoundError:
            st.caption("Run `python -m src.model.evaluate` to populate metrics.")


# ──────────────────────── HITL: review/approve/edit/reject ────────────────────────
if st.session_state.stage == "review":
    st.markdown("---")
    st.subheader("👤 Human-in-the-loop checkpoint")
    st.caption("Review the draft, edit if needed, then approve or reject.")

    edited = st.text_area(
        "Draft brief (editable)",
        value=st.session_state.draft,
        height=400,
        key="draft_editor",
    )

    col_a, col_b, col_c = st.columns(3)
    if col_a.button("✅ Approve as-is", type="primary"):
        st.session_state.final_brief = st.session_state.draft
        st.session_state.stage = "done"
        st.rerun()
    if col_b.button("📝 Save edits"):
        st.session_state.final_brief = edited
        st.session_state.stage = "done"
        st.rerun()
    if col_c.button("🔄 Reject & regenerate"):
        st.session_state.stage = "drafting"
        with st.status("Re-running orchestrator…", expanded=True):
            crew = build_crew(
                st.session_state.product_name,
                st.session_state.reviews_path,
                with_hitl=False,
            )
            st.session_state.draft = str(crew.kickoff())
        st.session_state.stage = "review"
        st.rerun()


# ──────────────────────── final brief + downloads ────────────────────────
if st.session_state.stage == "done" and st.session_state.final_brief:
    st.markdown("---")
    st.subheader("📄 Final market brief")

    out_path = cfg.output_dir / f"market_brief_{time.strftime('%Y%m%d_%H%M%S')}.md"
    if not out_path.exists():
        out_path.write_text(st.session_state.final_brief, encoding="utf-8")

    st.markdown(st.session_state.final_brief)

    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇️ Download brief (.md)",
        data=st.session_state.final_brief,
        file_name=out_path.name,
        mime="text/markdown",
    )
    if st.session_state.log_path and Path(st.session_state.log_path).exists():
        c2.download_button(
            "⬇️ Download JSON log",
            data=Path(st.session_state.log_path).read_bytes(),
            file_name=Path(st.session_state.log_path).name,
            mime="application/x-ndjson",
        )

    if c1.button("🆕 New run"):
        for k in ("stage", "draft", "final_brief", "log_path"):
            st.session_state[k] = "idle" if k == "stage" else ("" if isinstance(st.session_state[k], str) else None)
        st.rerun()


# ──────────────────────── live log tail ────────────────────────
if st.session_state.log_path and Path(st.session_state.log_path).exists():
    with st.expander("📋 Agent action log (JSONL)", expanded=False):
        lines = Path(st.session_state.log_path).read_text(encoding="utf-8").strip().splitlines()
        st.caption(f"{len(lines)} log lines · `{Path(st.session_state.log_path).name}`")
        st.code("\n".join(lines[-30:]) or "(no events yet)", language="json")
