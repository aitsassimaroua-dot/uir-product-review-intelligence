"""BertSentimentTool — wraps the fine-tuned BERT classifier as a CrewAI tool.

Behaviour:
- If a trained checkpoint exists at `cfg.model_dir`, use it (real path).
- Otherwise fall back to a deterministic keyword heuristic (W1 stub) so the
  agent loop is testable before W2 is finished.
"""
from __future__ import annotations

import hashlib
import time
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.config import cfg
from src.utils.logging_config import get_logger

log = get_logger("tool.sentiment")


class BertSentimentInput(BaseModel):
    text: str = Field(..., description="Review text to classify.")


_HEURISTIC_NEG = {"bad", "terrible", "awful", "hate", "broken", "useless", "worst", "disappointed", "poor", "waste"}
_HEURISTIC_POS = {"great", "love", "excellent", "amazing", "perfect", "best", "good", "fantastic", "happy"}


def _heuristic(text: str) -> dict:
    t = text.lower()
    pos = sum(w in t for w in _HEURISTIC_POS)
    neg = sum(w in t for w in _HEURISTIC_NEG)
    if neg > pos:
        return {"label": "negative", "score": min(0.5 + 0.1 * (neg - pos), 0.95), "backend": "heuristic"}
    return {"label": "positive", "score": min(0.5 + 0.1 * (pos - neg), 0.95), "backend": "heuristic"}


class BertSentimentTool(BaseTool):
    name: str = "bert_sentiment"
    description: str = (
        "Classify the sentiment of a single product review as 'positive' or 'negative'. "
        "Input: a review text string. Output: {label, score} where score is the model "
        "confidence in [0,1]. Use this tool one review at a time."
    )
    args_schema: Type[BaseModel] = BertSentimentInput

    def _run(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            log.warning("tool.invalid_input", extra={"tool": self.name, "reason": "empty_text"})
            return {"error": "empty_text", "label": None, "score": 0.0}

        input_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        t0 = time.time()
        try:
            result = self._classify(text)
        except Exception as exc:
            log.exception("tool.error", extra={"tool": self.name, "input_hash": input_hash})
            return {"error": str(exc), "label": None, "score": 0.0}

        latency_ms = round((time.time() - t0) * 1000, 1)
        log.info(
            "tool.call",
            extra={
                "tool": self.name,
                "input_hash": input_hash,
                "label": result["label"],
                "score": result["score"],
                "backend": result.get("backend", "bert"),
                "latency_ms": latency_ms,
            },
        )
        return result

    def _classify(self, text: str) -> dict:
        # Lazy import — keeps the agent runnable in W1 without torch/transformers.
        if (cfg.model_dir / "config.json").exists():
            from src.model.inference import predict  # local import on purpose
            label, score = predict(text)
            return {"label": label, "score": float(score), "backend": "bert"}
        return _heuristic(text)
