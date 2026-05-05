"""Market Researcher — owns the web search tool.

Given the product and the analyst's top complaints, finds competitors and
external context. Outputs a short list of competitors with evidence URLs.
"""
from __future__ import annotations

from crewai import Agent

from src.agents.llm import make_llm
from src.tools.competitor_search_tool import CompetitorSearchTool


def build_market_researcher() -> Agent:
    return Agent(
        role="Market Research Specialist",
        goal=(
            "Identify 3 plausible competitors for the product under review and gather "
            "1–2 short evidence snippets for each (with URLs). Use the complaints surfaced "
            "by the Sentiment Analyst to guide your search queries."
        ),
        backstory=(
            "You are a market analyst who relies on public web sources. You always cite the URL "
            "you got each claim from. If a search returns nothing, you say so explicitly rather "
            "than invent a competitor."
        ),
        tools=[CompetitorSearchTool()],
        llm=make_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=10,
    )
