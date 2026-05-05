"""ReviewLoaderTool — loads a CSV of reviews into a structured list.

Required columns: `text`. Optional: `id`, `rating`. Extra columns are kept
as metadata but truncated to 5 fields to avoid blowing up agent context.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils.logging_config import get_logger

log = get_logger("tool.review_loader")


class ReviewLoaderInput(BaseModel):
    path: str = Field(..., description="Filesystem path to a CSV with at least a `text` column.")
    max_rows: int = Field(default=200, ge=1, le=2000, description="Cap on rows returned.")


class ReviewLoaderTool(BaseTool):
    name: str = "review_loader"
    description: str = (
        "Load product reviews from a CSV file. Returns a list of {id, text, rating?} records. "
        "Required column: 'text'. Optional: 'id', 'rating'. Use once at the start of the analysis."
    )
    args_schema: Type[BaseModel] = ReviewLoaderInput

    def _run(self, path: str, max_rows: int = 200) -> list[dict] | dict:
        p = Path(path)
        if not p.exists():
            log.warning("tool.error", extra={"tool": self.name, "reason": "file_not_found", "path": str(p)})
            return {"error": f"File not found: {p}"}

        rows: list[dict] = []
        try:
            with p.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or "text" not in reader.fieldnames:
                    return {"error": "CSV must have a 'text' column."}
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    text = (row.get("text") or "").strip()
                    if not text:
                        continue
                    record = {
                        "id": int(row["id"]) if row.get("id", "").isdigit() else i,
                        "text": text,
                    }
                    if "rating" in row and row["rating"]:
                        try:
                            record["rating"] = int(row["rating"])
                        except ValueError:
                            pass
                    rows.append(record)
        except Exception as exc:
            log.exception("tool.error", extra={"tool": self.name, "path": str(p)})
            return {"error": str(exc)}

        log.info("tool.call", extra={"tool": self.name, "path": str(p), "rows_returned": len(rows)})
        return rows
