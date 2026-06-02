"""V0 Streamlit demo.

Renders one QueryResponse from the API. Surfaces, per the design doc:
  - the answer with inline [n] citation markers,
  - a faithfulness badge slot (shows the score when present; V0 ships with
    RAGAS faithfulness off by default, so it reads "—" until V1 turns it on),
  - a sources panel where CITED chunks get a colored badge and
    retrieved-but-not-cited chunks get a neutral one (same shape, different color),
  - a metrics row: latency breakdown, cost (flagged estimate), tokens, chunk count.

Talks to the FastAPI service over HTTP so the demo and engine stay decoupled.
Start the API first (uvicorn sec_rag.api.app:app), then `make demo`.
Set SEC_RAG_API_URL to point elsewhere (default http://localhost:8000).

No model selector: model choice is a dev/eval concern (design doc), not a UI knob.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("SEC_RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="sec-filings-rag", layout="wide")
st.title("SEC filings RAG")

query = st.text_input(
    "Question",
    placeholder="Ask about a 10-K — e.g., 'What were Apple's biggest risk factors in FY2023?'",
)

if st.button("Ask", type="primary") and query.strip():
    try:
        resp = httpx.post(f"{API_URL}/query", json={"query": query}, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # demo: show the failure plainly
        st.error(f"Request failed: {exc}")
        st.stop()

    metrics = data["metrics"]

    # Answer + faithfulness badge slot.
    faith = metrics.get("faithfulness")
    badge = f"Faithfulness {faith:.2f}" if faith is not None else "Faithfulness —"
    answer_col, badge_col = st.columns([5, 1])
    with answer_col:
        st.markdown("### Answer")
        st.write(data["answer"])
    with badge_col:
        st.metric("self-grade", badge)
        if faith is None:
            st.caption("RAGAS off in V0")

    # Sources: cited (colored) vs retrieved-but-not-cited (neutral).
    st.markdown("### Sources")
    for c in data["citations"]:
        tag = f":blue-background[[{c['source_index']}] cited]" if c["cited"] \
            else f":gray-background[{c['source_index']} retrieved]"
        head = f"{tag}  **{c['doc_name']}**"
        if c.get("section"):
            head += f" — {c['section']}"
        if c.get("page") is not None:
            head += f"  (p.{c['page']})"
        head += f"  · score {c['retrieval_score']:.2f}"
        st.markdown(head)
        st.caption(c["excerpt"][:600] + ("…" if len(c["excerpt"]) > 600 else ""))

    # Metrics row.
    st.markdown("### Metrics")
    cols = st.columns(5)
    cols[0].metric("latency (ms)", metrics["latency_ms"])
    cols[1].metric("retrieval (ms)", metrics["retrieval_ms"])
    cols[2].metric("generation (ms)", metrics["generation_ms"])
    cost_label = f"${metrics['cost_usd']:.4f}" + (" est" if metrics.get("cost_is_estimate") else "")
    cols[3].metric("cost", cost_label)
    cols[4].metric("chunks", metrics["chunks_retrieved"])
    st.caption(
        f"tokens in/out {metrics['tokens_in']}/{metrics['tokens_out']} · "
        f"model {data['model']} · trace {data['trace_id'][:8]}"
    )
