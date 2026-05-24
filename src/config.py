"""Centralized, env-driven configuration.

Single import surface — every module reads settings from `cfg`, never from
os.environ directly. Keeps test overrides simple and the rest of the code
free of magic strings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw else default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    # LLM
    model: str = _env("MODEL", "gemini/gemini-2.0-flash")
    gemini_api_key: str = _env("GEMINI_API_KEY", "")
    ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")

    # Paths (resolved from project root)
    model_dir: Path = field(default_factory=lambda: ROOT / _env("MODEL_DIR", "models/sentiment_bert"))
    log_dir: Path = field(default_factory=lambda: ROOT / _env("LOG_DIR", "logs"))
    output_dir: Path = field(default_factory=lambda: ROOT / _env("OUTPUT_DIR", "outputs"))
    data_dir: Path = field(default_factory=lambda: ROOT / "data")

    # Training
    hf_dataset: str = _env("HF_DATASET", "amazon_polarity")
    base_model: str = _env("BASE_MODEL", "distilbert-base-uncased")
    num_labels: int = _env_int("NUM_LABELS", 2)
    train_subset: int = _env_int("TRAIN_SUBSET", 50000)
    eval_subset: int = _env_int("EVAL_SUBSET", 5000)
    batch_size: int = _env_int("BATCH_SIZE", 16)
    learning_rate: float = _env_float("LEARNING_RATE", 2e-5)
    num_epochs: int = _env_int("NUM_EPOCHS", 2)
    max_length: int = _env_int("MAX_LENGTH", 256)

    # Runtime
    log_level: str = _env("LOG_LEVEL", "INFO")

    def label_names(self) -> list[str]:
        return ["negative", "positive"] if self.num_labels == 2 else [f"class_{i}" for i in range(self.num_labels)]


cfg = Settings()

# Make sure runtime dirs exist (cheap and idempotent).
for _p in (cfg.log_dir, cfg.output_dir, cfg.data_dir / "processed"):
    _p.mkdir(parents=True, exist_ok=True)
