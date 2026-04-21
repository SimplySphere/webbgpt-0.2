from __future__ import annotations

from html import escape

from config import ServeConfig


def render_playground_html(config: ServeConfig) -> str:
    model_name = escape(config.model_name)
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>WebbGPT Playground</title>
    <style>
      :root {{
        --bg: #f4efe4;
        --panel: rgba(255, 252, 246, 0.92);
        --panel-strong: #fffaf0;
        --panel-soft: rgba(255, 255, 255, 0.5);
        --ink: #20170f;
        --muted: #6b5d52;
        --line: rgba(53, 35, 18, 0.14);
        --accent: #b14d1f;
        --accent-strong: #8f3410;
        --accent-soft: rgba(177, 77, 31, 0.12);
        --user: #1f5f8b;
        --assistant: #5d3b8c;
        --good: #1f7a4d;
        --good-soft: rgba(31, 122, 77, 0.12);
        --warn: #8a5a11;
        --warn-soft: rgba(138, 90, 17, 0.12);
        --bad: #9b1c1c;
        --bad-soft: rgba(155, 28, 28, 0.12);
        --shadow: 0 24px 60px rgba(52, 31, 15, 0.12);
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(177, 77, 31, 0.14), transparent 28%),
          radial-gradient(circle at top right, rgba(31, 95, 139, 0.12), transparent 26%),
          linear-gradient(180deg, #f8f2e8 0%, var(--bg) 55%, #efe6d7 100%);
        min-height: 100vh;
      }}

      .shell {{
        width: min(1240px, calc(100vw - 32px));
        margin: 24px auto;
        display: grid;
        grid-template-columns: 320px 1fr;
        gap: 20px;
      }}

      .panel {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 24px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(14px);
      }}

      .sidebar {{
        padding: 24px;
        display: flex;
        flex-direction: column;
        gap: 18px;
      }}

      .eyebrow {{
        margin: 0;
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
        font-weight: 700;
      }}

      h1 {{
        margin: 0;
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 0.95;
      }}

      .lede {{
        margin: 0;
        color: var(--muted);
        line-height: 1.55;
      }}

      .badge-grid {{
        display: grid;
        gap: 10px;
      }}

      .badge {{
        padding: 12px 14px;
        border-radius: 16px;
        background: var(--panel-strong);
        border: 1px solid var(--line);
      }}

      .badge strong {{
        display: block;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 5px;
      }}

      .controls {{
        display: grid;
        gap: 12px;
      }}

      label.toggle {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        padding: 12px 14px;
        border-radius: 16px;
        border: 1px solid var(--line);
        background: var(--panel-strong);
        font-size: 0.98rem;
      }}

      label.toggle.disabled {{
        opacity: 0.6;
      }}

      .toggle input {{
        width: 18px;
        height: 18px;
      }}

      .control-note {{
        margin: -2px 2px 2px;
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.45;
      }}

      button {{
        border: 0;
        border-radius: 16px;
        padding: 12px 16px;
        font: inherit;
        cursor: pointer;
      }}

      .ghost {{
        background: transparent;
        border: 1px solid var(--line);
        color: var(--ink);
      }}

      .primary {{
        background: linear-gradient(135deg, var(--accent), var(--accent-strong));
        color: #fff8f3;
        font-weight: 700;
      }}

      .workspace {{
        padding: 22px;
        display: grid;
        gap: 16px;
      }}

      .topbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        padding: 4px 4px 0;
      }}

      .topbar h2 {{
        margin: 0;
        font-size: 1.05rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }}

      .chat-log {{
        min-height: 58vh;
        max-height: 58vh;
        overflow: auto;
        padding: 10px 4px 4px;
        display: grid;
        gap: 14px;
      }}

      .empty {{
        padding: 28px;
        border: 1px dashed var(--line);
        border-radius: 20px;
        color: var(--muted);
        background: rgba(255, 255, 255, 0.45);
      }}

      .message {{
        padding: 18px 18px 14px;
        border-radius: 22px;
        border: 1px solid var(--line);
        background: var(--panel-strong);
      }}

      .message.user {{
        border-left: 6px solid var(--user);
      }}

      .message.assistant {{
        border-left: 6px solid var(--assistant);
      }}

      .message .role {{
        margin: 0 0 10px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
      }}

      .message pre {{
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font: inherit;
        line-height: 1.65;
      }}

      .badge-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
      }}

      .pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: var(--panel-soft);
        font-size: 0.82rem;
        letter-spacing: 0.03em;
      }}

      .pill.good {{
        background: var(--good-soft);
        color: var(--good);
      }}

      .pill.warn {{
        background: var(--warn-soft);
        color: var(--warn);
      }}

      .pill.fail {{
        background: var(--bad-soft);
        color: var(--bad);
      }}

      .summary-line {{
        margin: 12px 0 0;
        color: var(--muted);
        line-height: 1.45;
      }}

      .citation-block {{
        margin-top: 10px;
        color: var(--muted);
      }}

      .citation-shell {{
        border: 0;
        background: transparent;
      }}

      .citation-shell summary {{
        cursor: pointer;
        list-style: none;
        font-size: 0.9rem;
        line-height: 1.4;
        color: var(--muted);
      }}

      .citation-shell summary::-webkit-details-marker {{
        display: none;
      }}

      .citation-empty {{
        font-size: 0.9rem;
        line-height: 1.4;
        color: var(--muted);
      }}

      .citation-groups {{
        display: grid;
        gap: 8px;
        margin-top: 8px;
      }}

      .citation-group {{
        border-left: 2px solid var(--line);
        padding-left: 10px;
      }}

      .citation-group summary {{
        cursor: pointer;
        list-style: none;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 4px 0;
        font-weight: 600;
      }}

      .citation-group summary::-webkit-details-marker {{
        display: none;
      }}

      .citation-group-count {{
        color: var(--muted);
        font-size: 0.88rem;
        font-weight: 400;
      }}

      .citation-snippets {{
        padding: 2px 0 8px;
        display: grid;
        gap: 10px;
      }}

      .citation-snippets p {{
        margin: 0;
        line-height: 1.5;
      }}

      .citation-detail-label {{
        color: var(--muted);
        font-size: 0.9rem;
      }}

      .divider {{
        margin-top: 14px;
        border-top: 1px solid var(--line);
      }}

      .trace-card {{
        margin-top: 14px;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.5);
        overflow: hidden;
      }}

      .trace-card[hidden] {{
        display: none;
      }}

      .trace-card summary {{
        cursor: pointer;
        list-style: none;
        padding: 14px 16px;
        font-weight: 700;
        color: var(--ink);
      }}

      .trace-card summary::-webkit-details-marker {{
        display: none;
      }}

      .trace-body {{
        padding: 0 16px 16px;
        display: grid;
        gap: 14px;
      }}

      .trace-section {{
        border-top: 1px solid var(--line);
        padding-top: 12px;
      }}

      .trace-section h4 {{
        margin: 0 0 8px;
        font-size: 0.84rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
      }}

      .kv-list {{
        display: grid;
        gap: 8px;
      }}

      .kv {{
        display: grid;
        grid-template-columns: 180px 1fr;
        gap: 12px;
        align-items: start;
        font-size: 0.95rem;
      }}

      .kv dt {{
        color: var(--muted);
      }}

      .kv dd {{
        margin: 0;
        word-break: break-word;
      }}

      .trace-actions {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }}

      .subtle-button {{
        padding: 8px 12px;
        border-radius: 12px;
        border: 1px solid var(--line);
        background: transparent;
      }}

      .timeline {{
        display: grid;
        gap: 8px;
      }}

      .timeline-item {{
        display: grid;
        grid-template-columns: 150px 1fr;
        gap: 10px;
        font-size: 0.94rem;
      }}

      .timeline-item strong {{
        color: var(--muted);
        font-weight: 600;
      }}

      .failure-box {{
        margin-top: 12px;
        padding: 14px 16px;
        border-radius: 18px;
        border: 1px solid rgba(155, 28, 28, 0.28);
        background: rgba(155, 28, 28, 0.06);
      }}

      .failure-box strong {{
        display: block;
        margin-bottom: 6px;
        color: var(--bad);
      }}

      .assistant-actions {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
      }}

      .assistant-actions-wrap {{
        margin-top: 12px;
      }}

      .composer {{
        display: grid;
        gap: 12px;
        padding-top: 10px;
        border-top: 1px solid var(--line);
      }}

      textarea {{
        width: 100%;
        min-height: 128px;
        resize: vertical;
        padding: 16px 18px;
        border-radius: 18px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.76);
        color: var(--ink);
        font: inherit;
        line-height: 1.55;
      }}

      .composer-actions {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
      }}

      .hint {{
        color: var(--muted);
        font-size: 0.92rem;
      }}

      .status {{
        min-height: 24px;
        color: var(--muted);
      }}

      .status.error {{
        color: var(--bad);
      }}

      code.inline {{
        padding: 1px 6px;
        border-radius: 999px;
        background: rgba(53, 35, 18, 0.06);
      }}

      @media (max-width: 900px) {{
        .shell {{
          grid-template-columns: 1fr;
        }}

        .chat-log {{
          min-height: 44vh;
          max-height: none;
        }}

        .composer-actions {{
          flex-direction: column;
          align-items: stretch;
        }}

        .kv,
        .timeline-item {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="shell">
      <aside class="panel sidebar">
        <p class="eyebrow">Local Playground</p>
        <h1>WebbGPT</h1>
        <p class="lede">
          Your AI assistant for all things Webb.
        </p>

        <div class="badge-grid">
          <section class="badge">
            <strong>Model</strong>
            <span>{model_name}</span>
          </section>
          <section class="badge">
            <strong>API</strong>
            <span><code>/v1/chat/completions</code></span>
          </section>
        </div>

        <div class="controls">
          <label class="toggle">
            <span>Tailor for Webb</span>
            <input id="toolsToggle" type="checkbox" checked />
          </label>
          <label id="citationsToggleLabel" class="toggle">
            <span>Show Citations</span>
            <input id="citationsToggle" type="checkbox" checked />
          </label>
          <div id="controlsNote" class="control-note">
            Show Citations adds source labels when Webb tailoring finds grounded sources.
          </div>
        </div>

        <button id="clearButton" class="ghost" type="button">New Conversation</button>
      </aside>

      <section class="panel workspace">
        <div class="topbar">
          <h2>Chat Session</h2>
          <div class="hint">Use Shift+Enter for a new line, Enter to send.</div>
        </div>

        <div id="chatLog" class="chat-log">
          <div class="empty">
            Start with a general question, or ask a catalog-style question like “What does AdvSt Chemistry require?” to test grounding.
          </div>
        </div>

        <form id="composer" class="composer">
          <textarea
            id="promptInput"
            placeholder="Ask WebbGPT anything..."
            aria-label="Prompt"
          ></textarea>
          <div class="composer-actions">
            <div id="statusLine" class="status"></div>
            <button id="sendButton" class="primary" type="submit">Send Prompt</button>
          </div>
        </form>
      </section>
    </main>

    <script>
      const messages = [];
      const chatLog = document.getElementById("chatLog");
      const composer = document.getElementById("composer");
      const promptInput = document.getElementById("promptInput");
      const sendButton = document.getElementById("sendButton");
      const clearButton = document.getElementById("clearButton");
      const statusLine = document.getElementById("statusLine");
      const toolsToggle = document.getElementById("toolsToggle");
      const citationsToggle = document.getElementById("citationsToggle");
      const citationsToggleLabel = document.getElementById("citationsToggleLabel");
      const controlsNote = document.getElementById("controlsNote");

      function setStatus(text, isError = false) {{
        statusLine.textContent = text;
        statusLine.classList.toggle("error", isError);
      }}

      function shortId(value) {{
        if (!value || typeof value !== "string") {{
          return "n/a";
        }}
        return value.slice(0, 12);
      }}

      function shortPath(value) {{
        if (!value || typeof value !== "string") {{
          return "n/a";
        }}
        const parts = value.split("/");
        return parts[parts.length - 1] || value;
      }}

      function createElement(tag, className, text) {{
        const element = document.createElement(tag);
        if (className) {{
          element.className = className;
        }}
        if (text !== undefined) {{
          element.textContent = text;
        }}
        return element;
      }}

      function addBadge(container, text, tone = "") {{
        const badge = createElement("span", "pill" + (tone ? " " + tone : ""), text);
        container.appendChild(badge);
      }}

      function extractMeta(item) {{
        return (item.meta && item.meta.metadata) || {{}};
      }}

      function extractStatus(item) {{
        return extractMeta(item).status || {{}};
      }}

      function extractProvenance(item) {{
        return extractMeta(item).provenance || {{}};
      }}

      function extractReproCapsule(item) {{
        return extractMeta(item).repro_capsule || {{}};
      }}

      function extractRequest(item) {{
        return (item.meta && item.meta.request) || {{}};
      }}

      function captureRequestState(conversation, safeDecode) {{
        const tailoringRequested = Boolean(toolsToggle.checked);
        const citationsSelected = Boolean(citationsToggle.checked);
        return {{
          messages: conversation.map(item => ({{ role: item.role, content: item.content }})),
          tools: tailoringRequested,
          citations: tailoringRequested && citationsSelected,
          citationsSelected,
          safe_decode: Boolean(safeDecode),
        }};
      }}

      function syncDependentControls() {{
        const tailoringRequested = Boolean(toolsToggle.checked);
        citationsToggle.disabled = !tailoringRequested;
        citationsToggleLabel.classList.toggle("disabled", !tailoringRequested);
        controlsNote.textContent = tailoringRequested
          ? "Show Citations adds source labels when Webb tailoring finds grounded sources."
          : "Show Citations requires Tailor for Webb, because citations only come from grounded Webb sources.";
      }}

      function fallbackCopyText(payload, successMessage) {{
        const textarea = document.createElement("textarea");
        textarea.value = payload;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        textarea.style.pointerEvents = "none";
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        textarea.setSelectionRange(0, textarea.value.length);
        try {{
          const copied = document.execCommand("copy");
          setStatus(copied ? (successMessage || "Copied.") : "Copy failed.", !copied);
        }} catch (_error) {{
          setStatus("Copy failed.", true);
        }} finally {{
          document.body.removeChild(textarea);
        }}
      }}

      function copyText(text, successMessage) {{
        const payload = typeof text === "string" ? text : JSON.stringify(text, null, 2);
        if (navigator.clipboard && window.isSecureContext) {{
          navigator.clipboard.writeText(payload)
            .then(() => setStatus(successMessage || "Copied."))
            .catch(() => fallbackCopyText(payload, successMessage));
          return;
        }}
        fallbackCopyText(payload, successMessage);
      }}

      function createKvSection(title, rows) {{
        const section = createElement("section", "trace-section");
        section.appendChild(createElement("h4", "", title));
        const list = createElement("dl", "kv-list");
        for (const row of rows) {{
          if (row.value === null || row.value === undefined || row.value === "") {{
            continue;
          }}
          const wrapper = createElement("div", "kv");
          const dt = createElement("dt", "", row.label);
          const dd = createElement("dd", "");
          dd.textContent = String(row.value);
          wrapper.append(dt, dd);
          list.appendChild(wrapper);
        }}
        if (!list.children.length) {{
          return null;
        }}
        section.appendChild(list);
        return section;
      }}

      function createTimelineSection(entries) {{
        const section = createElement("section", "trace-section");
        section.appendChild(createElement("h4", "", "Timeline"));
        const timeline = createElement("div", "timeline");
        for (const entry of entries || []) {{
          const row = createElement("div", "timeline-item");
          row.appendChild(createElement("strong", "", entry.label || ""));
          row.appendChild(createElement("span", "", entry.value || ""));
          timeline.appendChild(row);
        }}
        if (!timeline.children.length) {{
          return null;
        }}
        section.appendChild(timeline);
        return section;
      }}

      function requestPayload(conversation, safeDecode) {{
        const request = captureRequestState(conversation, safeDecode);
        return {{
          messages: request.messages,
          tools: request.tools,
          citations: request.citations,
          safe_decode: request.safe_decode,
        }};
      }}

      async function requestCompletion(conversation, safeDecode = false) {{
        const response = await fetch("/v1/chat/completions", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(requestPayload(conversation, safeDecode)),
        }});

        if (!response.ok) {{
          const text = await response.text();
          throw new Error(text || ("Request failed with status " + response.status));
        }}

        return response.json();
      }}

      function createTraceCard(item, index) {{
        const meta = extractMeta(item);
        const request = extractRequest(item);
        const provenance = extractProvenance(item);
        const reproCapsule = extractReproCapsule(item);
        const status = meta.status || {{}};
        const routing = meta.routing || {{}};
        const grounding = meta.grounding || {{}};
        const generation = meta.generation || {{}};
        const quality = meta.quality || {{}};
        const trace = createElement("details", "trace-card");
        trace.dataset.messageIndex = String(index);
        trace.hidden = true;
        const summary = createElement("summary", "", "Details");
        trace.appendChild(summary);

        const body = createElement("div", "trace-body");
        const debug = meta.debug || {{}};
        const decode = provenance.decode || {{}};
        const exportArtifact = provenance.export || {{}};
        const responseText = debug.raw_output || item.content || "";

        const responseStatus = createKvSection("Response Status", [
          {{ label: "Interpretation", value: meta.summary }},
          {{ label: "Grounded response", value: status.grounded ? "yes" : "no" }},
          {{ label: "Citations shown", value: status.cited ? "yes" : "no" }},
          {{ label: "Abstained", value: status.abstained ? "yes" : "no" }},
          {{ label: "Degenerate output", value: status.degenerate_output ? "yes" : "no" }},
          {{ label: "Webb tailoring used", value: item.meta.usedTools ? "yes" : "no" }},
          {{ label: "Citation count", value: (item.meta.citations || []).length }},
          {{ label: "Response characters", value: responseText.length }},
        ]);
        if (responseStatus) {{
          body.appendChild(responseStatus);
        }}

        const requestSection = createKvSection("Request Settings", [
          {{ label: "Tailor for Webb requested", value: request.tools ? "yes" : "no" }},
          {{ label: "Show Citations selected", value: request.citationsSelected ? "yes" : "no" }},
          {{ label: "Show Citations active", value: request.citations ? "yes" : "no" }},
          {{ label: "Safe decode requested", value: request.safeDecode ? "yes" : "no" }},
        ]);
        if (requestSection) {{
          body.appendChild(requestSection);
        }}

        const groundingSection = createKvSection("Grounding", [
          {{ label: "Routed as", value: routing.mode || "chat" }},
          {{ label: "Catalog queried", value: routing.catalog_queried ? "yes" : "no" }},
          {{ label: "Retrieved hits", value: grounding.retrieved_hits ?? 0 }},
          {{ label: "Abstained on no hits", value: grounding.abstained_due_to_no_hits ? "yes" : "no" }},
          {{ label: "Citation labels", value: (grounding.citation_labels || []).join(", ") || "none" }},
          {{ label: "Snapshot label", value: grounding.catalog_snapshot_label || "n/a" }},
        ]);
        if (groundingSection) {{
          body.appendChild(groundingSection);
        }}

        const checkpoint = provenance.checkpoint || {{}};
        const tokenizer = provenance.tokenizer || {{}};
        const modelArtifact = createKvSection("Model Artifact", [
          {{ label: "Checkpoint", value: shortPath(checkpoint.path) }},
          {{ label: "Checkpoint path", value: checkpoint.path }},
          {{ label: "Checkpoint id", value: shortId(checkpoint.artifact_id) }},
          {{ label: "Checkpoint sha256", value: checkpoint.checkpoint_sha256 }},
          {{ label: "Directory sha256", value: checkpoint.directory_sha256 }},
          {{ label: "Tokenizer", value: shortPath(tokenizer.path) }},
          {{ label: "Tokenizer path", value: tokenizer.path }},
          {{ label: "Tokenizer id", value: shortId(tokenizer.artifact_id) }},
          {{ label: "Tokenizer sha256", value: tokenizer.sha256 }},
          {{ label: "Tokenizer vocab sha256", value: tokenizer.vocab_sha256 || tokenizer.tokenizer_vocab_sha256 }},
          {{ label: "Tokenizer config sha256", value: tokenizer.metadata_sha256 || tokenizer.tokenizer_config_sha256 }},
          {{ label: "Export path", value: exportArtifact.path }},
          {{ label: "Export id", value: shortId(exportArtifact.artifact_id) }},
          {{ label: "Export directory sha256", value: exportArtifact.directory_sha256 }},
        ]);
        if (modelArtifact) {{
          body.appendChild(modelArtifact);
        }}

        const snapshot = provenance.catalog_snapshot || {{}};
        const snapshotSection = createKvSection("Catalog Snapshot", [
          {{ label: "Snapshot label", value: grounding.catalog_snapshot_label || shortPath(snapshot.sqlite_path || snapshot.catalog_input_path) }},
          {{ label: "Snapshot id", value: shortId(snapshot.snapshot_id) }},
          {{ label: "Catalog DSN", value: snapshot.catalog_dsn }},
          {{ label: "Catalog input", value: snapshot.catalog_input_path }},
          {{ label: "Catalog input sha256", value: snapshot.catalog_input_sha256 }},
          {{ label: "SQLite path", value: snapshot.sqlite_path }},
          {{ label: "SQLite sha256", value: snapshot.sqlite_sha256 }},
        ]);
        if (snapshotSection) {{
          body.appendChild(snapshotSection);
        }}

        const decodeSection = createKvSection("Decode Settings", [
          {{ label: "Backend", value: generation.backend }},
          {{ label: "Preset", value: generation.decode_preset }},
          {{ label: "Safe decode", value: generation.safe_decode ? "yes" : "no" }},
          {{ label: "Stop reason", value: generation.stop_reason }},
          {{ label: "Max new tokens", value: decode.max_new_tokens }},
          {{ label: "Temperature", value: decode.temperature }},
          {{ label: "Top p", value: decode.top_p }},
          {{ label: "Repetition penalty", value: decode.repetition_penalty }},
          {{ label: "No-repeat ngram", value: decode.no_repeat_ngram_size }},
          {{ label: "Stop strings", value: (decode.stop_strings || []).join(", ") || "none" }},
        ]);
        if (decodeSection) {{
          body.appendChild(decodeSection);
        }}

        const seeds = reproCapsule.seed_bundle || {{}};
        const reproducibility = createKvSection("Reproducibility", [
          {{ label: "Checkpoint id", value: shortId(reproCapsule.checkpoint_artifact_id) }},
          {{ label: "Tokenizer id", value: shortId(reproCapsule.tokenizer_artifact_id) }},
          {{ label: "Snapshot id", value: shortId(reproCapsule.snapshot_id) }},
          {{ label: "Backend", value: reproCapsule.backend }},
          {{ label: "Decode preset", value: reproCapsule.decode_preset }},
          {{ label: "Seed bundle", value: "python=" + (seeds.python ?? "n/a") + ", numpy=" + (seeds.numpy ?? "n/a") + ", torch=" + (seeds.torch ?? "n/a") }},
        ]);
        if (reproducibility) {{
          body.appendChild(reproducibility);
        }}

        if (quality.degenerate || (quality.reasons && quality.reasons.length)) {{
          const qualitySection = createKvSection("Response Quality", [
            {{ label: "Degenerate", value: quality.degenerate ? "yes" : "no" }},
            {{ label: "Reasons", value: (quality.reasons || []).join(", ") || "none" }},
            {{ label: "Token count", value: quality.metrics?.token_count }},
            {{ label: "Alpha ratio", value: quality.metrics?.alpha_ratio }},
            {{ label: "Comma ratio", value: quality.metrics?.comma_ratio }},
            {{ label: "Separator bursts", value: quality.metrics?.separator_bursts }},
            {{ label: "Short fragment ratio", value: quality.metrics?.short_fragment_ratio }},
            {{ label: "Punctuation suffix ratio", value: quality.metrics?.punctuation_suffix_ratio }},
            {{ label: "Unique token ratio", value: quality.metrics?.unique_token_ratio }},
            {{ label: "Repeated token run", value: quality.metrics?.repeated_token_run }},
            {{ label: "Non-space chars", value: quality.metrics?.nonspace_chars }},
          ]);
          if (qualitySection) {{
            body.appendChild(qualitySection);
          }}
        }}

        const timeline = createTimelineSection(meta.timeline || []);
        if (timeline) {{
          body.appendChild(timeline);
        }}

        const rawOutputSection = createKvSection("Raw Output", [
          {{ label: "Output", value: responseText }},
        ]);
        if (rawOutputSection) {{
          body.appendChild(rawOutputSection);
        }}

        trace.appendChild(body);
        return trace;
      }}

      function createAssistantActions(index, item) {{
        const meta = extractMeta(item);
        const status = meta.status || {{}};
        if (!status.degenerate_output || index !== messages.length - 1) {{
          return null;
        }}
        const wrapper = createElement("div", "assistant-actions-wrap");
        const badgeRow = createElement("div", "badge-row");
        addBadge(badgeRow, "Malformed output", "fail");
        wrapper.appendChild(badgeRow);
        const actions = createElement("div", "assistant-actions");
        const reproCapsule = extractReproCapsule(item);

        const retry = createElement("button", "ghost", "Retry");
        retry.type = "button";
        retry.addEventListener("click", async () => {{
          await retryAssistant(index, false);
        }});
        actions.appendChild(retry);

        const safeRetry = createElement("button", "ghost", "Retry With Safe Preset");
        safeRetry.type = "button";
        safeRetry.addEventListener("click", async () => {{
          await retryAssistant(index, true);
        }});
        actions.appendChild(safeRetry);

        const copyCapsule = createElement("button", "ghost", "Copy Repro Capsule");
        copyCapsule.type = "button";
        copyCapsule.addEventListener("click", () => {{
          copyText(reproCapsule, "Repro capsule copied.");
        }});
        actions.appendChild(copyCapsule);

        const copyDebug = createElement("button", "ghost", "Copy Full Debug JSON");
        copyDebug.type = "button";
        copyDebug.addEventListener("click", () => {{
          copyText(meta, "Full debug JSON copied.");
        }});
        actions.appendChild(copyDebug);

        const showDetails = createElement("button", "ghost", "Show Details");
        showDetails.type = "button";
        showDetails.addEventListener("click", () => {{
          const trace = chatLog.querySelector(`.trace-card[data-message-index="${{index}}"]`);
          if (!trace) {{
            return;
          }}
          const shouldShow = trace.hidden;
          trace.hidden = !shouldShow ? true : false;
          if (shouldShow) {{
            trace.open = true;
            trace.scrollIntoView({{ behavior: "smooth", block: "nearest" }});
            showDetails.textContent = "Hide Details";
          }} else {{
            trace.open = false;
            showDetails.textContent = "Show Details";
          }}
        }});
        actions.appendChild(showDetails);

        wrapper.appendChild(actions);
        return wrapper;
      }}

      function createVisibleCitationBlock(item) {{
        const meta = extractMeta(item);
        const request = extractRequest(item);
        const status = meta.status || {{}};
        const grounding = meta.grounding || {{}};
        const rawCitations = (item.meta && item.meta.citations) || [];
        const groups = new Map();

        function simplifyLabel(label) {{
          const text = String(label || "").trim();
          if (!text) {{
            return "Source";
          }}
          const parts = text.split("|").map(part => part.trim()).filter(Boolean);
          if (!parts.length) {{
            return text;
          }}
          if (parts.length > 1 && /The Webb Schools|Private Boarding & Day School in California/i.test(parts.slice(1).join(" | "))) {{
            return parts[0];
          }}
          return text;
        }}

        function cleanSnippet(snippet, label) {{
          let text = String(snippet || "").trim();
          if (!text) {{
            return "";
          }}
          const shortLabel = simplifyLabel(label).toLowerCase();
          const lines = text.split(/\\n+/).map(part => part.trim()).filter(Boolean);
          if (lines.length > 1 && simplifyLabel(lines[0]).toLowerCase() === shortLabel) {{
            text = lines.slice(1).join("\\n").trim();
          }}
          const lowered = text.toLowerCase();
          const prefix = shortLabel + " ";
          if (lowered.startsWith(prefix)) {{
            text = text.slice(prefix.length).trim();
          }}
          return text;
        }}

        function ensureGroup(rawLabel) {{
          const displayLabel = simplifyLabel(rawLabel);
          if (!groups.has(displayLabel)) {{
            groups.set(displayLabel, {{
              displayLabel,
              rawLabels: new Set(),
              snippets: [],
              snippetKeys: new Set(),
            }});
          }}
          const group = groups.get(displayLabel);
          group.rawLabels.add(String(rawLabel || displayLabel).trim());
          return group;
        }}

        for (const citation of rawCitations) {{
          const rawLabel = citation.label || citation.source_id || citation.source_type || "Source";
          const snippet = cleanSnippet(citation.snippet || "", rawLabel);
          const group = ensureGroup(rawLabel);
          const snippetKey = snippet;
          if (snippet && !group.snippetKeys.has(snippetKey)) {{
            group.snippetKeys.add(snippetKey);
            group.snippets.push(snippet);
          }}
        }}

        if (!groups.size) {{
          for (const rawLabel of grounding.citation_labels || []) {{
            ensureGroup(rawLabel);
          }}
        }}

        if (!groups.size) {{
          if (!request.citations) {{
            return null;
          }}
          const block = createElement("section", "citation-block");
          let note = "Sources: none surfaced for this response.";
          if (!status.grounded) {{
            note = "Sources: none. This response was model-only.";
          }} else if (grounding.abstained_due_to_no_hits) {{
            note = "Sources: none found in the current Webb snapshot.";
          }} else if (status.degenerate_output) {{
            note = "Sources: none surfaced because the response was intercepted as malformed.";
          }}
          block.appendChild(createElement("div", "citation-empty", note));
          return block;
        }}

        const block = createElement("section", "citation-block");
        const shell = createElement("details", "citation-shell");
        const shellSummary = createElement(
          "summary",
          "",
          "Sources: " + Array.from(groups.keys()).join(", ")
        );
        shell.appendChild(shellSummary);
        const list = createElement("div", "citation-groups");
        for (const group of groups.values()) {{
          const detail = createElement("details", "citation-group");
          const summary = createElement("summary", "");
          summary.appendChild(createElement("span", "", group.displayLabel));
          const countText = group.snippets.length
            ? `${{group.snippets.length}} passage${{group.snippets.length === 1 ? "" : "s"}}`
            : `${{group.rawLabels.size}} source${{group.rawLabels.size === 1 ? "" : "s"}}`;
          summary.appendChild(createElement("span", "citation-group-count", countText));
          detail.appendChild(summary);

          const snippets = createElement("div", "citation-snippets");
          if (group.snippets.length) {{
            group.snippets.forEach(snippet => {{
              snippets.appendChild(createElement("p", "", snippet));
            }});
          }} else {{
            Array.from(group.rawLabels).forEach(rawLabel => {{
              snippets.appendChild(createElement("p", "citation-detail-label", rawLabel));
            }});
          }}
          detail.appendChild(snippets);
          list.appendChild(detail);
        }}
        shell.appendChild(list);
        block.appendChild(shell);
        return block;
      }}

      function render() {{
        chatLog.innerHTML = "";
        if (!messages.length) {{
          chatLog.innerHTML = '<div class="empty">Start with a general question, or ask a catalog-style question like “What does AdvSt Chemistry require?” to test grounding.</div>';
          return;
        }}

        messages.forEach((item, index) => {{
          const card = createElement("article", "message " + item.role);
          const role = createElement("p", "role", item.role);
          card.appendChild(role);

          if (item.role === "assistant") {{
            const meta = extractMeta(item);
            const status = meta.status || {{}};
            const debug = meta.debug || {{}};
            const body = createElement("pre", "");
            body.textContent = status.degenerate_output && debug.raw_output ? debug.raw_output : item.content;
            card.appendChild(body);

            const citationBlock = createVisibleCitationBlock(item);
            if (citationBlock) {{
              card.appendChild(citationBlock);
            }}

            const badges = createElement("div", "badge-row");
            if (!status.degenerate_output) {{
              addBadge(badges, status.grounded ? "Grounded" : "Model Only", status.grounded ? "good" : "warn");
              if (status.grounded) {{
                addBadge(badges, "Catalog", "good");
              }}
              if (status.cited) {{
                addBadge(badges, "Cited", "good");
              }}
              addBadge(badges, status.abstained ? "Abstained" : "Answered", status.abstained ? "warn" : "");
            }}
            card.appendChild(badges);

            if (meta.summary && !status.degenerate_output) {{
              card.appendChild(createElement("p", "summary-line", meta.summary));
            }}

            const actions = createAssistantActions(index, item);
            if (actions) {{
              card.appendChild(actions);
            }}

            if (status.degenerate_output) {{
              card.appendChild(createElement("div", "divider"));
              card.appendChild(createTraceCard(item, index));
            }}
          }} else {{
            const body = createElement("pre", "");
            body.textContent = item.content;
            card.appendChild(body);
          }}

          chatLog.appendChild(card);
        }});

        chatLog.scrollTop = chatLog.scrollHeight;
      }}

      function normalizeAssistantPayload(payload, request) {{
        return {{
          role: "assistant",
          content: payload.text || "",
          meta: {{
            usedTools: payload.used_tools,
            citations: payload.citations || [],
            request: {{
              tools: Boolean(request && request.tools),
              citations: Boolean(request && request.citations),
              citationsSelected: Boolean(request && request.citationsSelected),
              safeDecode: Boolean(request && request.safe_decode),
            }},
            metadata: payload.metadata || {{}},
          }},
        }};
      }}

      async function sendPrompt() {{
        const prompt = promptInput.value.trim();
        if (!prompt) {{
          setStatus("Enter a prompt first.", true);
          return;
        }}

        messages.push({{ role: "user", content: prompt }});
        render();
        promptInput.value = "";
        sendButton.disabled = true;
        setStatus("Generating response...");

        try {{
          const request = captureRequestState(messages, false);
          const payload = await requestCompletion(messages, false);
          messages.push(normalizeAssistantPayload(payload, request));
          render();
          setStatus("Response ready.");
        }} catch (error) {{
          setStatus(error.message || "Request failed.", true);
        }} finally {{
          sendButton.disabled = false;
        }}
      }}

      async function retryAssistant(index, safeDecode) {{
        const history = messages.slice(0, index);
        if (!history.length) {{
          return;
        }}
        messages.splice(index);
        render();
        sendButton.disabled = true;
        setStatus(safeDecode ? "Retrying with safe preset..." : "Retrying response...");
        try {{
          const request = captureRequestState(history, safeDecode);
          const payload = await requestCompletion(history, safeDecode);
          messages.push(normalizeAssistantPayload(payload, request));
          render();
          setStatus(safeDecode ? "Safe retry complete." : "Retry complete.");
        }} catch (error) {{
          setStatus(error.message || "Retry failed.", true);
        }} finally {{
          sendButton.disabled = false;
        }}
      }}

      composer.addEventListener("submit", async event => {{
        event.preventDefault();
        await sendPrompt();
      }});

      promptInput.addEventListener("keydown", async event => {{
        if (event.key === "Enter" && !event.shiftKey) {{
          event.preventDefault();
          await sendPrompt();
        }}
      }});

      clearButton.addEventListener("click", () => {{
        messages.length = 0;
        render();
        setStatus("Conversation cleared.");
        promptInput.focus();
      }});

      toolsToggle.addEventListener("change", () => {{
        syncDependentControls();
      }});

      render();
      syncDependentControls();
      promptInput.focus();
    </script>
  </body>
</html>
"""
