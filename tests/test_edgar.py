"""EDGAR offline logic (no network): HTML/table extraction, ticker, form/multi."""

import pytest

from sec_rag.edgar.client import html_to_text, resolve_ticker
from sec_rag.edgar.live_engine import detect_form, detect_multi


def test_html_to_text_strips_tags_and_scripts():
    html = """
    <html><head><style>.x{color:red}</style><script>var a=1;</script></head>
    <body><h1>Item 7.</h1><p>Net sales were $32,765 million.</p></body></html>
    """
    out = html_to_text(html)
    assert "Net sales were $32,765 million." in out
    assert "var a" not in out and "color:red" not in out  # script/style dropped


def test_resolve_unknown_ticker_raises():
    # Force the cache so no network call is needed.
    import sec_rag.edgar.client as c
    c._ticker_map = {"AAPL": ("0000320193", "Apple Inc.")}
    assert resolve_ticker("aapl") == ("0000320193", "Apple Inc.")  # case-insensitive
    with pytest.raises(ValueError):
        resolve_ticker("NOTATICKER")


def test_table_extractor_merges_split_currency_cells():
    # SEC tables split "$", number, and ")" into separate <td>s.
    html = """<table>
      <tr><td>Total net sales</td><td>$</td><td>3,306</td></tr>
      <tr><td>Operating loss</td><td>$</td><td>(1,234</td><td>)</td></tr>
    </table>"""
    out = html_to_text(html)
    assert "Total net sales | $3,306" in out
    assert "Operating loss | $(1,234)" in out


def test_detect_form():
    assert detect_form("What was total revenue last year?") == "10-K"
    assert detect_form("revenue in the most recent quarter?") == "10-Q"
    assert detect_form("summarize the latest 8-K announcement") == "8-K"


def test_detect_multi():
    assert detect_multi("What was revenue last year?") == 1
    assert detect_multi("Compare revenue this year vs last year") == 2
    assert detect_multi("Revenue trend over the last 3 years") == 3
