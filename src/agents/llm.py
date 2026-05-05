"""LLM factory — single source of truth for the model passed to each agent.

CrewAI uses LiteLLM under the hood, so the model string follows LiteLLM
routing (e.g. `gemini/gemini-1.5-flash`, `ollama/llama3.1`).
"""
from __future__ import annotations

from crewai import LLM

from src.config import cfg


def make_llm() -> LLM:
    kwargs: dict = {"model": cfg.model}
    if cfg.model.startswith("gemini/"):
        if not cfg.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is empty. Set it in .env (https://aistudio.google.com/apikey)."
            )
        kwargs["api_key"] = cfg.gemini_api_key
    elif cfg.model.startswith("ollama/"):
        kwargs["base_url"] = cfg.ollama_base_url
    return LLM(**kwargs)
