"""Sentiment Analyst — owns the BERT tool.

Reads reviews, classifies each, surfaces recurring themes. Output is plain
prose the orchestrator can splice into the brief, but anchored on numeric
counts so the orchestrator can't hallucinate the headline figure.
"""
from __future__ import annotations

from crewai import Agent

from src.agents.llm import make_llm
from src.tools.review_loader_tool import ReviewLoaderTool
from src.tools.sentiment_tool import BertSentimentTool


def build_sentiment_analyst() -> Agent:
    return Agent(
        role="Senior Sentiment Analyst",
        goal=(
            "Classify every review as positive or negative using the BERT tool, then "
            "report (1) the exact counts of each label, (2) the top 3 recurring complaints, "
            "(3) the top 3 recurring praises. Always cite which reviews you used."
        ),
        backstory=(
            "You are a sentiment analyst who never trusts your gut over the model. "
            "You always call the bert_sentiment tool — you never label reviews on your own. "
            "You quote reviews verbatim when surfacing themes."
        ),
        tools=[ReviewLoaderTool(), BertSentimentTool()],
        llm=make_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=15,
    )
