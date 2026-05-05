"""JSON structured logging.

Every record is one JSON object per line (JSONL) — easy to grep, parse,
and feed into a dashboard later. Required by the brief: "every agent
action logged with timestamps (JSON)".
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import cfg


class JsonLineFormatter(logging.Formatter):
    """One-line JSON record per log entry."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Anything passed via logger.info("x", extra={"agent": "..."}) gets surfaced.
        for key, value in record.__dict__.items():
            if key in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "module", "msecs",
                "message", "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName", "taskName",
            }:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = repr(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(run_id: str | None = None) -> Path:
    """Wire root logger: JSON to file (JSONL) + human-readable to stderr.

    Returns the path of the JSONL file so callers can reference it (e.g.
    print it at end of run for the user).
    """
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    log_path = cfg.log_dir / f"agent_actions_{run_id}.jsonl"

    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))
    # Reset handlers (idempotent — useful in notebooks/tests).
    root.handlers.clear()

    file_h = logging.FileHandler(log_path, encoding="utf-8")
    file_h.setFormatter(JsonLineFormatter())
    root.addHandler(file_h)

    stderr_h = logging.StreamHandler(sys.stderr)
    stderr_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(stderr_h)

    root.info("logging.configured", extra={"log_path": str(log_path), "run_id": run_id})
    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
