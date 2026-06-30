"""EDGAR client offline logic (no network): HTML extraction + unknown ticker."""

import pytest

from sec_rag.edgar.client import html_to_text, resolve_ticker


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
