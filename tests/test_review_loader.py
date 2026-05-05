"""Review loader tests — input validation and error paths."""
from pathlib import Path

import pytest

from src.tools.review_loader_tool import ReviewLoaderTool


def test_missing_file_returns_error_dict(tmp_path):
    tool = ReviewLoaderTool()
    out = tool._run(path=str(tmp_path / "nope.csv"))
    assert isinstance(out, dict) and "error" in out


def test_missing_text_column_returns_error(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("id,foo\n1,hello\n")
    out = ReviewLoaderTool()._run(path=str(p))
    assert isinstance(out, dict) and "error" in out


def test_loads_valid_csv(tmp_path):
    p = tmp_path / "good.csv"
    p.write_text("id,text,rating\n1,Great product,5\n2,Awful,1\n")
    out = ReviewLoaderTool()._run(path=str(p))
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["text"] == "Great product"
    assert out[0]["rating"] == 5


def test_skips_empty_text_rows(tmp_path):
    p = tmp_path / "withempty.csv"
    p.write_text("id,text\n1,\n2,Real review\n")
    out = ReviewLoaderTool()._run(path=str(p))
    assert len(out) == 1
    assert out[0]["text"] == "Real review"


def test_max_rows_cap(tmp_path):
    p = tmp_path / "many.csv"
    rows = "id,text\n" + "\n".join(f"{i},review {i}" for i in range(50))
    p.write_text(rows)
    out = ReviewLoaderTool()._run(path=str(p), max_rows=10)
    assert len(out) == 10
