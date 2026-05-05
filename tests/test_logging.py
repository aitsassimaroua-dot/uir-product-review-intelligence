"""JSON logging tests — every record is a valid JSON line and includes the key fields."""
import json
import logging
from pathlib import Path

import src.config as _config
from src.utils.logging_config import configure_logging


def test_records_are_valid_jsonl(tmp_path):
    # cfg is a frozen dataclass: bypass via object.__setattr__, restore on exit.
    original = _config.cfg.log_dir
    object.__setattr__(_config.cfg, "log_dir", tmp_path)
    try:
        log_path = configure_logging(run_id="test_jsonl")
        logging.getLogger("test").info("event.x", extra={"agent": "tester", "value": 42})
        for h in logging.getLogger().handlers:
            h.flush()
        lines = Path(log_path).read_text(encoding="utf-8").strip().splitlines()
        assert lines, "expected at least one log line"
        parsed = [json.loads(l) for l in lines]
        assert all("ts" in r and "level" in r and "msg" in r for r in parsed)
        assert any(r.get("agent") == "tester" and r.get("value") == 42 for r in parsed)
    finally:
        # Detach file handler so the tmp dir can be cleaned up cleanly.
        for h in list(logging.getLogger().handlers):
            logging.getLogger().removeHandler(h)
            h.close()
        object.__setattr__(_config.cfg, "log_dir", original)
