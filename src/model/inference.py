"""Single-shot + batched inference API for the fine-tuned BERT.

Loaded lazily and cached at module level — calling `predict()` repeatedly
re-uses the same in-memory model.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import torch

from src.config import cfg


@lru_cache(maxsize=1)
def _load():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_dir)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    id2label = model.config.id2label or {0: "negative", 1: "positive"}
    return tokenizer, model, device, id2label


def predict(text: str) -> Tuple[str, float]:
    tokenizer, model, device, id2label = _load()
    inputs = tokenizer(
        text, truncation=True, max_length=cfg.max_length, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    idx = int(torch.argmax(probs).item())
    return str(id2label[idx]), float(probs[idx].item())


def predict_batch(texts: List[str], batch_size: int = 32) -> List[Tuple[str, float]]:
    tokenizer, model, device, id2label = _load()
    out: List[Tuple[str, float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        inputs = tokenizer(
            chunk, truncation=True, padding=True, max_length=cfg.max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        idxs = torch.argmax(probs, dim=-1).cpu().tolist()
        scores = probs.max(dim=-1).values.cpu().tolist()
        out.extend((str(id2label[i]), float(s)) for i, s in zip(idxs, scores))
    return out
