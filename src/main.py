"""CLI entry point.

Usage:
    python -m src.main --product "Wireless Earbuds X" --reviews data/processed/sample_reviews.csv

Wires logging, builds the crew, runs it, writes the approved brief to disk.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.config import cfg
from src.crew import build_crew
from src.utils.logging_config import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Product Review Intelligence — multi-agent run")
    p.add_argument("--product", required=True, help="Product name being analyzed.")
    p.add_argument("--reviews", required=True, help="Path to a CSV with at least a `text` column.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = configure_logging(run_id)
    log = get_logger("main")

    reviews_path = Path(args.reviews)
    if not reviews_path.exists():
        log.error("input.missing", extra={"path": str(reviews_path)})
        print(f"ERROR: reviews file not found: {reviews_path}", file=sys.stderr)
        return 2

    log.info("run.start", extra={"run_id": run_id, "product": args.product, "reviews_path": str(reviews_path)})
    crew = build_crew(args.product, str(reviews_path))

    try:
        result = crew.kickoff()
    except Exception:
        log.exception("run.failed", extra={"run_id": run_id})
        print("ERROR: run failed. See logs for details.", file=sys.stderr)
        return 1

    brief = str(result)
    out_path = cfg.output_dir / f"market_brief_{run_id}.md"
    out_path.write_text(brief, encoding="utf-8")

    log.info("run.complete", extra={"run_id": run_id, "brief_path": str(out_path), "log_path": str(log_path)})
    print(f"\n✓ Brief written to {out_path}")
    print(f"✓ JSON action log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
