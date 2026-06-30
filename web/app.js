// SEC Filings RAG — static frontend.
// Calls the deployed FastAPI service's /query/stream (SSE) and renders the
// streamed answer + sources + metrics. No API key / settings: the API is open
// for now (auth is added back later).

const BUILD = "v10 · live-edgar";

const IS_LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);
const API = IS_LOCAL
  ? "http://localhost:8000"
  : "https://sec-rag-api-200217758117.us-east1.run.app";

const $ = (id) => document.getElementById(id);

// --- Render helpers ---
function linkifyCitations(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML.replace(/\[(\d+)\]/g, '<span class="cite">[$1]</span>');
}

function renderSources(citations) {
  const box = $("sources");
  box.innerHTML = "";
  for (const c of citations) {
    const el = document.createElement("div");
    el.className = "source";
    const tag = c.cited
      ? `<span class="tag cited">[${c.source_index}] cited</span>`
      : `<span class="tag retr">${c.source_index} retrieved</span>`;
    const meta = [c.section, c.page != null ? `p.${c.page}` : null, `score ${(c.retrieval_score ?? 0).toFixed(2)}`]
      .filter(Boolean).join(" · ");
    const excerpt = (c.excerpt || "").slice(0, 320);
    el.innerHTML =
      `<div>${tag}<span class="doc">${c.doc_name || ""}</span></div>` +
      `<div class="meta">${meta}</div>` +
      `<div class="excerpt">${excerpt}${(c.excerpt || "").length > 320 ? "…" : ""}</div>`;
    box.appendChild(el);
  }
}

function renderMetrics(m, model) {
  const cost = m.cost_usd != null ? `$${m.cost_usd.toFixed(4)}` : "—";
  $("metrics").innerHTML =
    `<span>latency <b>${m.latency_ms} ms</b></span>` +
    `<span>retrieval <b>${m.retrieval_ms} ms</b></span>` +
    `<span>generation <b>${m.generation_ms} ms</b></span>` +
    `<span>cost <b>${cost}</b></span>` +
    `<span>tokens <b>${m.tokens_in}/${m.tokens_out}</b></span>` +
    `<span>chunks <b>${m.chunks_retrieved}</b></span>` +
    `<span>model <b>${model}</b></span>`;
}

// --- Ask (streaming) ---
let busy = false;

// A ticker routes to the live EDGAR path (any company's latest 10-K); blank uses
// the pre-indexed FinanceBench corpus. Both stream the same event shape.
async function ask(question, ticker) {
  if (busy || !question.trim()) return;
  busy = true;
  const live = !!(ticker && ticker.trim());

  $("result").hidden = false;
  $("error").hidden = true;
  $("error").textContent = "";
  $("status").hidden = true;
  $("status").textContent = "";
  $("answer").innerHTML = live
    ? `Fetching ${ticker.toUpperCase()}'s latest 10-K from EDGAR + indexing… (~15–30s the first time)`
    : "Thinking… (first request after idle can take ~15–25s)";
  $("sources").innerHTML = "";
  $("metrics").innerHTML = "";
  $("faithBadge").textContent = "…";
  $("askBtn").disabled = true;
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });

  const url = live ? API + "/query/live/stream" : API + "/query/stream";
  const body = live ? { ticker: ticker.trim(), query: question } : { query: question };

  let answerText = "";
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop();
      for (const block of blocks) {
        const line = block.trim();
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        const ev = JSON.parse(payload);
        if (ev.type === "status") {
          $("status").hidden = false;
          $("status").textContent = "📄 " + ev.text;
        } else if (ev.type === "token") {
          answerText += ev.text;
          $("answer").innerHTML = linkifyCitations(answerText);
        } else if (ev.type === "done") {
          const r = ev.response;
          $("answer").innerHTML = linkifyCitations(r.answer);
          renderSources(r.citations || []);
          renderMetrics(r.metrics, r.model);
          const f = r.metrics.faithfulness;
          $("faithBadge").textContent = f != null ? `Faithfulness ${f.toFixed(2)}` : "Faithfulness —";
        } else if (ev.type === "error") {
          throw new Error(ev.detail);
        }
      }
    }
  } catch (e) {
    $("error").hidden = false;
    $("error").textContent = "Request failed: " + e.message;
    $("answer").innerHTML = "";
    $("faithBadge").textContent = "—";
  } finally {
    busy = false;
    $("askBtn").disabled = false;
  }
}

// --- Wire up ---
$("askForm").addEventListener("submit", (e) => {
  e.preventDefault();
  ask($("queryInput").value, $("tickerInput").value);
});
$("chips").addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) {
    const ticker = e.target.dataset.ticker || "";
    $("tickerInput").value = ticker;
    $("queryInput").value = e.target.textContent;
    ask(e.target.textContent, ticker);
  }
});

const _b = $("build");
if (_b) _b.textContent = "build " + BUILD + (IS_LOCAL ? " · local" : "");
