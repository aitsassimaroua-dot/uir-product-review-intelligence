"""CompetitorSearchTool — DuckDuckGo web search.

Free, no API key. Returns top results with title, snippet, url. Wrapped so a
network failure or rate-limit returns a structured error instead of crashing
the agent loop.
"""
from __future__ import annotations

import time
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils.logging_config import get_logger

log = get_logger("tool.competitor_search")


class CompetitorSearchInput(BaseModel):
    query: str = Field(..., min_length=2, description="Search query, e.g. 'wireless earbuds X alternatives'.")
    k: int = Field(default=5, ge=1, le=10, description="Number of results to return.")


class CompetitorSearchTool(BaseTool):
    name: str = "competitor_search"
    description: str = (
        "Search the public web (DuckDuckGo) for competitor information about a product or feature. "
        "Returns up to k results, each with title, snippet, url. Use this to find competitor names, "
        "alternative products, or external reviews."
    )
    args_schema: Type[BaseModel] = CompetitorSearchInput

    def _run(self, query: str, k: int = 5) -> list[dict] | dict:
        # Lazy import: keeps the package import cheap and isolates the network dep.
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return {"error": "duckduckgo_search not installed; pip install duckduckgo-search"}

        t0 = time.time()
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=k))
        except Exception as exc:
            log.exception("tool.error", extra={"tool": self.name, "query": query})
            return {"error": f"search_failed: {exc}"}

        cleaned = [
            {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
            for r in results
        ][:k]

        latency_ms = round((time.time() - t0) * 1000, 1)
        log.info(
            "tool.call",
            extra={
                "tool": self.name,
                "query": query,
                "k": k,
                "n_results": len(cleaned),
                "latency_ms": latency_ms,
            },
        )
        if not cleaned:
            return {"warning": "no_results", "results": []}
        return cleaned
