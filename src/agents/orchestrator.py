"""Report Orchestrator — coordinator + writer.

Consumes the two specialists' outputs and writes a 1-page market brief.
Doesn't call BERT or web search itself — that separation is enforced by
giving it no tools.
"""
from __future__ import annotations

from crewai import Agent

from src.agents.llm import make_llm


def build_orchestrator() -> Agent:
    return Agent(
        role="Market Intelligence Orchestrator",
        goal=(
            "Synthesize the Sentiment Analyst's review summary and the Market Researcher's "
            "competitor scan into a single 1-page market brief: headline, 3-bullet sentiment "
            "summary, 3-bullet competitor signals, and a recommended action. Be concise and "
            "never invent numbers — only quote figures the analyst gave you."
        ),
        backstory=(
            "You are an editor, not a researcher. You write tight briefs for time-poor product "
            "managers. You refuse to fabricate data: if a section has no input, you say so."
        ),
        tools=[],  # synthesis only
        llm=make_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=8,
    )
