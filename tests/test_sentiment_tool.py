"""Sentiment tool tests — covers stub mode (no model on disk) and error paths.

Real-model tests live in test_inference.py and only run when the checkpoint exists.
"""
from src.tools.sentiment_tool import BertSentimentTool


def test_empty_input_returns_error_not_crash():
    tool = BertSentimentTool()
    out = tool._run(text="")
    assert out["error"] == "empty_text"
    assert out["score"] == 0.0


def test_whitespace_only_input_returns_error():
    tool = BertSentimentTool()
    out = tool._run(text="    \n\t  ")
    assert out["error"] == "empty_text"


def test_heuristic_picks_negative_on_negative_keywords():
    tool = BertSentimentTool()
    out = tool._run(text="This is awful and terrible, I hate it.")
    assert out["label"] == "negative"
    assert 0.0 < out["score"] <= 1.0


def test_heuristic_picks_positive_on_positive_keywords():
    tool = BertSentimentTool()
    out = tool._run(text="Absolutely love it, perfect and amazing.")
    assert out["label"] == "positive"


def test_output_schema_has_label_and_score():
    tool = BertSentimentTool()
    out = tool._run(text="Some random text here.")
    assert "label" in out
    assert "score" in out
    assert isinstance(out["score"], float)
