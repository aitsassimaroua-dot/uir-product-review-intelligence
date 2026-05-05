"""Crew assembly — builds the 3 agents and 3 tasks, returns a Crew.

Task chain (sequential):
  1. analyze_reviews   — Sentiment Analyst classifies the CSV
  2. scout_competitors — Market Researcher (uses 1's output as context)
  3. write_brief       — Orchestrator (HITL: human reviews before approval)
"""
from __future__ import annotations

from crewai import Crew, Process, Task

from src.agents import build_market_researcher, build_orchestrator, build_sentiment_analyst


def build_crew(product_name: str, reviews_path: str) -> Crew:
    analyst = build_sentiment_analyst()
    researcher = build_market_researcher()
    orchestrator = build_orchestrator()

    analyze_reviews = Task(
        description=(
            f"Load reviews from `{reviews_path}` using the review_loader tool, then call "
            f"bert_sentiment on EACH review (one at a time). Aggregate results.\n\n"
            f"Product under review: **{product_name}**\n\n"
            "Return a JSON object with keys:\n"
            "  - total_reviews (int)\n"
            "  - positive_count (int)\n"
            "  - negative_count (int)\n"
            "  - top_complaints (list of 3 short strings, drawn from negative reviews)\n"
            "  - top_praises (list of 3 short strings, drawn from positive reviews)\n"
            "  - sample_quotes (list of up to 4 verbatim quotes)\n"
        ),
        expected_output="A JSON object with the keys listed above.",
        agent=analyst,
    )

    scout_competitors = Task(
        description=(
            f"Using the complaints surfaced by the Sentiment Analyst, run 1–3 web searches "
            f"with the competitor_search tool to find 3 plausible competitors of `{product_name}`.\n\n"
            "Return a JSON object with keys:\n"
            "  - competitors: list of {name, why_relevant, evidence_url}\n"
            "  - market_signals: list of 2–3 short observations about the category\n"
        ),
        expected_output="A JSON object with `competitors` and `market_signals`.",
        agent=researcher,
        context=[analyze_reviews],  # researcher sees analyst output
    )

    write_brief = Task(
        description=(
            f"Write a 1-page market brief for **{product_name}** using ONLY the data above. "
            "Structure:\n"
            "  # Market Brief — {product_name}\n"
            "  ## Headline (one sentence)\n"
            "  ## Sentiment summary (3 bullets, with the exact pos/neg counts)\n"
            "  ## Competitive landscape (3 bullets, one per competitor, with URL)\n"
            "  ## Recommended next action (one sentence)\n\n"
            "Markdown only. No invented numbers."
        ),
        expected_output="A markdown brief following the structure above.",
        agent=orchestrator,
        context=[analyze_reviews, scout_competitors],
        human_input=True,  # <-- HITL checkpoint required by brief
        output_file=None,
    )

    return Crew(
        agents=[analyst, researcher, orchestrator],
        tasks=[analyze_reviews, scout_competitors, write_brief],
        process=Process.sequential,
        verbose=True,
    )
