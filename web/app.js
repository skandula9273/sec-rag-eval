// SEC Filings RAG — static frontend.
// Calls the deployed FastAPI service's /query/stream (SSE) and renders the
// streamed answer + sources + metrics. The API access key lives only in this
// browser's localStorage and is sent as the X-API-Key header.

// Local preview auto-points at the local API (where the X-API-Key guard is off);
// the deployed Pages site uses the Cloud Run API (guard on -> needs a key).
const IS_LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);
const DEFAULT_API = IS_LOCAL
  ? "http://localhost:8000"
  : "https://sec-rag-api-200217758117.us-east1.run.app";

const $ = (id) => document.getElementById(id);
const getKey = () => localStorage.getItem("secrag_key") || "";
const getApi = () => localStorage.getItem("secrag_api") || DEFAULT_API;

// --- Settings modal ---
function openModal() {
  $("keyInput").value = getKey();
  $("urlInput").value = getApi();
  $("modal").hidden = false;
}
function closeModal() { $("modal").hidden = true; }

$("settingsBtn").onclick = openModal;
$("settingsBtn2").onclick = openModal;
$("closeModal").onclick = closeModal;
$("saveKey").onclick = () => {
  localStorage.setItem("secrag_key", $("keyInput").value.trim());
  const url = $("urlInput").value.trim();
  localStorage.setItem("secrag_api", url || DEFAULT_API);
  closeModal();
};

// --- Render helpers ---
function linkifyCitations(text) {
  // Wrap [n] citation markers so they stand out in the answer.
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

async function ask(question) {
  if (busy || !question.trim()) return;
  if (!getKey() && !IS_LOCAL) { openModal(); return; }  // local API guard is off
  busy = true;

  $("result").hidden = false;
  $("error").hidden = true;
  $("error").textContent = "";
  $("answer").innerHTML = "";
  $("sources").innerHTML = "";
  $("metrics").innerHTML = "";
  $("faithBadge").textContent = "…";
  $("askBtn").disabled = true;
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });

  let answerText = "";
  try {
    const res = await fetch(getApi() + "/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": getKey() },
      body: JSON.stringify({ query: question }),
    });
    if (!res.ok) {
      const detail = res.status === 401 ? "invalid or missing API key" : `HTTP ${res.status}`;
      throw new Error(detail);
    }

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
        if (ev.type === "token") {
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
    $("faithBadge").textContent = "—";
  } finally {
    busy = false;
    $("askBtn").disabled = false;
  }
}

// --- Wire up ---
$("askForm").addEventListener("submit", (e) => {
  e.preventDefault();
  ask($("queryInput").value);
});
$("chips").addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) {
    $("queryInput").value = e.target.textContent;
    ask(e.target.textContent);
  }
});
