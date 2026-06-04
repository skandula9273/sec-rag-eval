"""QueryRequest validation — top_k bounds.

These guard the API contract boundary: bad top_k must fail here (422), not deep
in pgvector (LIMIT must not be negative) or the generation call (413 oversized
prompt). See api/schemas.py for the rationale.
"""

import pytest
from pydantic import ValidationError

from sec_rag.api.schemas import QueryRequest


def test_top_k_none_allowed():
    # None -> falls back to config default downstream.
    assert QueryRequest(query="x").top_k is None


def test_top_k_valid_range():
    assert QueryRequest(query="x", top_k=1).top_k == 1
    assert QueryRequest(query="x", top_k=5).top_k == 5
    assert QueryRequest(query="x", top_k=50).top_k == 50


@pytest.mark.parametrize("bad", [0, -1, -3, 51, 99999])
def test_top_k_out_of_bounds_rejected(bad):
    with pytest.raises(ValidationError):
        QueryRequest(query="x", top_k=bad)
