"""Edge-case scenarios required by the brief: 'edge cases tested, honest failure analysis'.

These tests document the expected graceful behaviour. They never crash; they
must return either the right answer, an explicit warning, or an error dict.
"""
import pytest

from src.tools.review_loader_tool import ReviewLoaderTool
from src.tools.sentiment_tool import BertSentimentTool


def test_extremely_long_text_does_not_crash():
    """A 10k-char review should still classify (heuristic doesn't truncate)."""
    out = BertSentimentTool()._run(text="this is great " * 1000)
    assert out["label"] in {"positive", "negative"}


def test_non_english_text_returns_a_label_anyway():
    """Heuristic mode is keyword-based and English-biased — we surface this in the report.
    Real BERT will also be English-biased on amazon_polarity. The contract is: it must
    not crash. Accuracy on non-English is documented as a known limitation."""
    out = BertSentimentTool()._run(text="Ce produit est génial, je l'adore !")
    assert "label" in out and "score" in out


def test_review_with_only_emoji():
    out = BertSentimentTool()._run(text="👍👍👍")
    assert "label" in out  # default fallthrough — positive in heuristic, BERT may differ


def test_csv_with_unicode_text(tmp_path):
    p = tmp_path / "unicode.csv"
    p.write_text("id,text\n1,Très bon produit\n2,日本語のレビュー\n", encoding="utf-8")
    out = ReviewLoaderTool()._run(path=str(p))
    assert isinstance(out, list)
    assert len(out) == 2


def test_csv_with_quoted_commas(tmp_path):
    p = tmp_path / "quoted.csv"
    p.write_text('id,text\n1,"Great, but pricey"\n', encoding="utf-8")
    out = ReviewLoaderTool()._run(path=str(p))
    assert len(out) == 1
    assert "pricey" in out[0]["text"]


@pytest.mark.parametrize("bad_input", [None, 123, [], {}])
def test_sentiment_tool_with_non_string_returns_error(bad_input):
    out = BertSentimentTool()._run(text=bad_input)
    assert "error" in out or out.get("label") is None
