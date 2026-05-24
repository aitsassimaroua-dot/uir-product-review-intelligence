"""BertSentimentTool — wraps the fine-tuned BERT classifier as a CrewAI tool.

Two call modes:
- Single:  text="…"           → {label, score}
- Batch:   texts=["…", "…"]   → [{label, score}, …]   (preferred, more reliable)

If a trained checkpoint exists at `cfg.model_dir` it is used; otherwise the
tool falls back to a deterministic keyword heuristic so the agent loop stays
testable when the model is absent.
"""
from __future__ import annotations

import hashlib
import time
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.config import cfg
from src.utils.logging_config import get_logger

log = get_logger("tool.sentiment")


class BertSentimentInput(BaseModel):
    text: Optional[str] = Field(
        default=None,
        description="A single review text to classify. Use for one-off calls.",
    )
    texts: Optional[list[str]] = Field(
        default=None,
        description=(
            "A list of review texts to classify in one call. "
            "STRONGLY PREFERRED when you have multiple reviews — far more reliable "
            "than calling the tool once per review."
        ),
    )


_HEURISTIC_NEG = {"bad", "terrible", "awful", "hate", "broken", "useless", "worst", "disappointed", "poor", "waste"}
_HEURISTIC_POS = {"great", "love", "excellent", "amazing", "perfect", "best", "good", "fantastic", "happy"}


def _heuristic(text: str) -> dict:
    t = text.lower()
    pos = sum(w in t for w in _HEURISTIC_POS)
    neg = sum(w in t for w in _HEURISTIC_NEG)
    if neg > pos:
        return {"label": "negative", "score": min(0.5 + 0.1 * (neg - pos), 0.95), "backend": "heuristic"}
    return {"label": "positive", "score": min(0.5 + 0.1 * (pos - neg), 0.95), "backend": "heuristic"}


def _has_real_model() -> bool:
    return (cfg.model_dir / "config.json").exists()


class BertSentimentTool(BaseTool):
    name: str = "bert_sentiment"
    description: str = (
        "Classify the sentiment of product reviews as 'positive' or 'negative' using a fine-tuned DistilBERT. "
        "Two modes: pass `text` for a single review, or pass `texts` (list of strings) for a batch — "
        "the batch mode is strongly recommended whenever you have more than one review, it is far more reliable "
        "than looping single calls. Output: {label, score} for single, or a list of {label, score} for batch."
    )
    args_schema: Type[BaseModel] = BertSentimentInput

    def _run(
        self,
        text: Optional[str] = None,
        texts: Optional[list[str]] = None,
    ) -> dict | list[dict]:
        if texts is not None:
            return self._run_batch(texts)
        if text is not None:
            return self._run_single(text)
        log.warning("tool.invalid_input", extra={"tool": self.name, "reason": "no_text_provided"})
        return {"error": "Must provide either 'text' (single) or 'texts' (list).", "label": None, "score": 0.0}

    def _summarise(self, predictions: list[dict], n_total: int) -> dict:
        pos = sum(1 for r in predictions if r.get("label") == "positive")
        neg = sum(1 for r in predictions if r.get("label") == "negative")
        return {
            "predictions": predictions,
            "summary": {
                "total_reviews": n_total,
                "positive_count": pos,
                "negative_count": neg,
            },
            "_instruction": (
                "AUTHORITATIVE COUNTS — copy these numbers verbatim. "
                "Do NOT recount the reviews yourself; trust the model."
            ),
        }

    # ────────────────── single ──────────────────
    def _run_single(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            log.warning("tool.invalid_input", extra={"tool": self.name, "reason": "empty_text"})
            return {"error": "empty_text", "label": None, "score": 0.0}

        input_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        t0 = time.time()
        try:
            result = self._classify_one(text)
        except Exception as exc:
            log.exception("tool.error", extra={"tool": self.name, "input_hash": input_hash})
            return {"error": str(exc), "label": None, "score": 0.0}

        latency_ms = round((time.time() - t0) * 1000, 1)
        log.info(
            "tool.call",
            extra={
                "tool": self.name,
                "mode": "single",
                "input_hash": input_hash,
                "label": result["label"],
                "score": result["score"],
                "backend": result.get("backend", "bert"),
                "latency_ms": latency_ms,
            },
        )
        return result

    # ────────────────── batch ──────────────────
    def _run_batch(self, texts: list[str]) -> dict:
        if not isinstance(texts, list) or not texts:
            log.warning("tool.invalid_input", extra={"tool": self.name, "reason": "empty_batch"})
            return {"error": "empty_batch", "predictions": [], "summary": {"total_reviews": 0, "positive_count": 0, "negative_count": 0}}

        # Filter empties but keep index alignment with placeholders.
        cleaned: list[tuple[int, str]] = [(i, t) for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
        if not cleaned:
            return {
                "predictions": [{"error": "empty_text", "label": None, "score": 0.0} for _ in texts],
                "summary": {"total_reviews": len(texts), "positive_count": 0, "negative_count": 0},
                "_instruction": "All inputs were empty.",
            }

        idxs = [i for i, _ in cleaned]
        valid_texts = [t for _, t in cleaned]

        t0 = time.time()
        try:
            results = self._classify_batch(valid_texts)
        except Exception as exc:
            log.exception("tool.error", extra={"tool": self.name, "n": len(texts)})
            return {
                "error": str(exc),
                "predictions": [{"error": str(exc), "label": None, "score": 0.0} for _ in texts],
                "summary": {"total_reviews": len(texts), "positive_count": 0, "negative_count": 0},
            }
        latency_ms = round((time.time() - t0) * 1000, 1)

        # Stitch back with placeholders for the empties.
        predictions: list[dict] = [{"error": "empty_text", "label": None, "score": 0.0} for _ in texts]
        for k, r in zip(idxs, results):
            predictions[k] = r

        out = self._summarise(predictions, len(texts))
        backend = results[0].get("backend", "bert") if results else "bert"
        log.info(
            "tool.call",
            extra={
                "tool": self.name,
                "mode": "batch",
                "n": len(texts),
                "positive": out["summary"]["positive_count"],
                "negative": out["summary"]["negative_count"],
                "backend": backend,
                "latency_ms": latency_ms,
            },
        )
        return out

    # ────────────────── inference ──────────────────
    def _classify_one(self, text: str) -> dict:
        if _has_real_model():
            from src.model.inference import predict
            label, score = predict(text)
            return {"label": label, "score": float(score), "backend": "bert"}
        return _heuristic(text)

    def _classify_batch(self, texts: list[str]) -> list[dict]:
        if _has_real_model():
            from src.model.inference import predict_batch
            preds = predict_batch(texts)
            return [{"label": lbl, "score": float(sc), "backend": "bert"} for lbl, sc in preds]
        return [_heuristic(t) for t in texts]
