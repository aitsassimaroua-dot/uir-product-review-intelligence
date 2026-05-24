"""Crew assembly — builds the 3 agents and 3 tasks, returns a Crew.

Task chain (sequential):
  1. analyze_reviews   — Sentiment Analyst classifies the CSV
  2. scout_competitors — Market Researcher (uses 1's output as context)
  3. write_brief       — Orchestrator (HITL: human reviews before approval)
"""
from __future__ import annotations

from crewai import Crew, Process, Task

from src.agents import build_market_researcher, build_orchestrator, build_sentiment_analyst


def build_crew(
    product_name: str,
    reviews_path: str,
    with_hitl: bool = True,
    feedback: str = "",
) -> Crew:
    """Build the 3-agent Crew.

    `with_hitl=True` uses CrewAI's terminal `human_input=True` (CLI demo).
    `with_hitl=False` skips it — the Streamlit UI handles approval externally
    via approve/edit/reject buttons.

    `feedback` is reviewer instructions injected into the orchestrator's task
    description on a re-run (Streamlit's "Revise with instructions" button).
    """
    analyst = build_sentiment_analyst()
    researcher = build_market_researcher()
    orchestrator = build_orchestrator()

    analyze_reviews = Task(
        description=(
            f"Product under review: **{product_name}**.\n\n"
            "Follow these steps EXACTLY in this order:\n"
            f"  1. Call `review_loader` once with path=`{reviews_path}` to get the list of reviews.\n"
            "  2. Extract every `text` field from the loaded records into a single Python list.\n"
            "  3. Call `bert_sentiment` ONCE with `texts=[the entire list of review texts]`. "
            "     The tool will return a JSON with `predictions`, `summary`, and `_instruction`.\n"
            "  4. **CRITICAL**: copy the numbers from `summary` VERBATIM into your output. "
            "     `total_reviews`, `positive_count`, `negative_count` MUST be the exact values from `summary`. "
            "     DO NOT recount the reviews yourself — the DistilBERT classifier is the authoritative source.\n"
            "  5. Pair each prediction in `predictions` with its original review text (by list index). "
            "     From reviews labelled 'negative', extract up to 3 recurring complaints. "
            "     From reviews labelled 'positive', extract up to 3 recurring praises.\n\n"
            "Return a JSON object with keys:\n"
            "  - total_reviews (int — copy from bert_sentiment summary)\n"
            "  - positive_count (int — copy from bert_sentiment summary)\n"
            "  - negative_count (int — copy from bert_sentiment summary)\n"
            "  - top_complaints (list of 3 short strings)\n"
            "  - top_praises (list of 3 short strings)\n"
            "  - sample_quotes (list of up to 4 verbatim quotes)\n\n"
            "Sanity check: positive_count + negative_count MUST equal total_reviews "
            "AND must equal the exact values from the tool's summary."
        ),
        expected_output="A JSON object with the keys listed above. Counts must be copied verbatim from bert_sentiment's summary field.",
        agent=analyst,
    )

    scout_competitors = Task(
        description=(
            f"Using the complaints surfaced by the Sentiment Analyst, run 1–3 web searches "
            f"with the competitor_search tool to find 3 plausible competitors of `{product_name}`.\n\n"
            "For each competitor you keep, you MUST have a real URL from the search results — "
            "pick the most relevant result's `url` field and store it verbatim in `evidence_url`. "
            "If no search result is plausible, drop that competitor; do NOT fabricate a URL.\n\n"
            "Return a JSON object with keys:\n"
            "  - competitors: list of objects, each with:\n"
            "      • name (string — the competitor's product/brand name)\n"
            "      • why_relevant (string — one short sentence)\n"
            "      • evidence_url (string — a real https:// URL copied from competitor_search results, "
            "        e.g. \"https://www.cnet.com/products/...\")\n"
            "  - market_signals: list of 2–3 short observations about the category\n"
        ),
        expected_output="A JSON object with `competitors` (each entry having a real URL in evidence_url) and `market_signals`.",
        agent=researcher,
        context=[analyze_reviews],  # researcher sees analyst output
    )

    brief_description = (
        f"Write a 1-page market brief for **{product_name}** using ONLY the data above.\n\n"
        "**STRICT RULES:**\n"
        "- The Sentiment Analyst's JSON output is the SOLE source of truth for counts. "
        "Copy `total_reviews`, `positive_count`, and `negative_count` VERBATIM. "
        "Do NOT recount the reviews yourself.\n"
        "- The Market Researcher's competitor list is the SOLE source of truth for competitors. "
        "Do NOT invent competitor names.\n"
        "- For each competitor, the Researcher provides an `evidence_url` field — that is a JSON KEY "
        "containing an actual HTTPS URL as its value (e.g. \"https://www.example.com/article\"). "
        "Paste that **literal URL string** into the markdown. NEVER write `{evidence_url}` or `#{evidence_url}` "
        "or any placeholder syntax — those are field NAMES, not values to template.\n\n"
        "Structure (exact):\n"
        f"  # Market Brief — {product_name}\n"
        "  ## Headline (one sentence)\n"
        "  ## Sentiment summary (two bullets ONLY — first bullet starts with the ACTUAL "
        "positive_count integer from the Analyst, second bullet starts with the ACTUAL "
        "negative_count integer. Example format: '- 8 positive reviews highlight ...' "
        "(replace 8 with the real number). Never write the letters X or Y.)\n"
        "  ## Competitive landscape (3 bullets — one per competitor — using this exact bullet format:\n"
        "      `- **Competitor name**: one-sentence reason. [Source](https://actual-url-here.com)`)\n"
        "  ## Recommended next action (one sentence)\n\n"
        "Markdown only. No invented numbers, no invented competitors, no template placeholders in the final text. "
        "Do NOT write 'X positive reviews' or 'Y negative reviews' — write the actual integer counts."
    )
    if feedback.strip():
        brief_description += (
            "\n\n**REVIEWER INSTRUCTIONS — apply these to your brief:**\n"
            f"{feedback.strip()}\n"
            "Incorporate these instructions while keeping the structure above. "
            "Do not invent new numbers or competitors — only use the data already provided."
        )

    write_brief = Task(
        description=brief_description,
        expected_output="A markdown brief following the structure above.",
        agent=orchestrator,
        context=[analyze_reviews, scout_competitors],
        human_input=with_hitl,  # CLI: prompt on stdin. UI: handled externally.
        output_file=None,
    )

    return Crew(
        agents=[analyst, researcher, orchestrator],
        tasks=[analyze_reviews, scout_competitors, write_brief],
        process=Process.sequential,
        verbose=True,
    )


def build_revision_crew(
    product_name: str,
    analyst_output: str,
    researcher_output: str,
    current_draft: str,
    feedback: str,
) -> Crew:
    """Single-task crew that only re-runs the orchestrator.

    Used when the user clicks 'Revise with these instructions' in the HITL:
    no need to re-classify reviews or re-search competitors — the specialists'
    outputs are passed inline in the task description.
    """
    orchestrator = build_orchestrator()

    revise_task = Task(
        description=(
            f"**TASK: Rewrite a market brief by applying a specific reviewer instruction.**\n\n"
            f"**Reviewer instruction (apply this NOW, do not return the same text):**\n"
            f"→ {feedback.strip()}\n\n"
            f"Previous draft for **{product_name}**:\n"
            f"--- BEGIN PREVIOUS DRAFT ---\n{current_draft}\n--- END PREVIOUS DRAFT ---\n\n"
            "Source data (use these numbers, never invent new ones):\n\n"
            f"--- SENTIMENT ANALYST OUTPUT ---\n{analyst_output}\n--- END ---\n\n"
            f"--- MARKET RESEARCHER OUTPUT ---\n{researcher_output}\n--- END ---\n\n"
            "**Rules:**\n"
            "- You MUST produce a NEW version that visibly applies the reviewer's instruction. "
            "Do NOT return the previous draft unchanged.\n"
            "- Keep the same section headings (Headline, Sentiment summary, Competitive landscape, Recommended next action).\n"
            "- Use the ACTUAL positive_count and negative_count integers from the Analyst output — "
            "never write the letters X or Y.\n"
            "- Keep the same competitors and their URLs from the Researcher output.\n"
            "- Output markdown only, no preamble."
        ),
        expected_output="A revised markdown brief incorporating the reviewer's instruction.",
        agent=orchestrator,
        human_input=False,
        output_file=None,
    )

    return Crew(
        agents=[orchestrator],
        tasks=[revise_task],
        process=Process.sequential,
        verbose=True,
    )
