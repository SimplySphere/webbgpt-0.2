from __future__ import annotations

import json
from html import escape
from pathlib import Path

from config import ServeConfig


def _checkpoint_label(path: str) -> str:
    name = Path(path).name
    parent = Path(path).parent.name
    return f"{parent}/{name}" if parent else name


def render_playground_html(config: ServeConfig) -> str:
    examples_json = json.dumps(
        [
            "hi WebbGPT 0.2, how are you?",
            "What is the difference between a prerequisite and a recommendation?",
            "What does a course catalog help students understand?",
            "What is the phone policy in the dining hall?",
            "A course catalog helps students",
            "During a science project, the first step is",
        ]
    )
    replacements = {
        "__MODEL_NAME__": escape(config.model_name),
        "__MODEL_MODE__": escape(config.model_mode),
        "__CHECKPOINT_PATH__": escape(config.checkpoint_path),
        "__CHECKPOINT_LABEL__": escape(_checkpoint_label(config.checkpoint_path)),
        "__RAG_MODE__": "on" if config.use_rag else "off",
        "__MAX_NEW_TOKENS__": str(int(config.max_new_tokens)),
        "__TEMPERATURE__": str(float(config.temperature)),
        "__TOP_K__": "" if config.top_k is None else str(int(config.top_k)),
        "__TOP_P__": str(float(config.top_p)),
        "__EXAMPLES_JSON__": examples_json,
    }
    html = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>WebbGPT 0.2 Chat</title>
    <style>
      :root {
        --bg: #f7f7f5;
        --surface: #ffffff;
        --surface-soft: #f1f4f2;
        --surface-warm: #fbf8f2;
        --text: #171717;
        --muted: #646a73;
        --line: #dfe4e1;
        --line-strong: #cdd5d0;
        --accent: #256f5c;
        --accent-dark: #15483b;
        --accent-soft: #e4f1ed;
        --user: #243b53;
        --assistant: #8a4b20;
        --warn: #8a5a11;
        --danger: #9b1c1c;
        --shadow: 0 18px 60px rgba(24, 32, 38, 0.08);
      }

      * {
        box-sizing: border-box;
      }

      html,
      body {
        height: 100%;
      }

      body {
        margin: 0;
        font-family:
          Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
          "Segoe UI", sans-serif;
        color: var(--text);
        background:
          linear-gradient(180deg, rgba(255, 255, 255, 0.78), rgba(247, 247, 245, 0.98)),
          var(--bg);
      }

      button,
      textarea,
      input {
        font: inherit;
      }

      button {
        border: 0;
        cursor: pointer;
      }

      .app {
        min-height: 100vh;
        display: grid;
        grid-template-rows: auto 1fr auto;
      }

      .topbar {
        position: sticky;
        top: 0;
        z-index: 10;
        border-bottom: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.86);
        backdrop-filter: blur(16px);
      }

      .topbar-inner {
        width: min(1040px, calc(100vw - 28px));
        min-height: 62px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
      }

      .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        min-width: 0;
      }

      .mark {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        display: grid;
        place-items: center;
        color: #fff;
        background: linear-gradient(135deg, var(--accent), var(--assistant));
        font-weight: 800;
        letter-spacing: 0.02em;
      }

      .brand-title {
        display: grid;
        gap: 2px;
        min-width: 0;
      }

      .brand-title strong {
        font-size: 1rem;
        line-height: 1.1;
      }

      .brand-title span {
        color: var(--muted);
        font-size: 0.82rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: min(42vw, 460px);
      }

      .status-strip {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 8px;
        flex-wrap: wrap;
      }

      .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        min-height: 28px;
        padding: 5px 9px;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: var(--surface);
        color: var(--muted);
        font-size: 0.78rem;
        white-space: nowrap;
      }

      .badge.good {
        color: var(--accent-dark);
        background: var(--accent-soft);
        border-color: #c7e2d9;
      }

      .badge.warn {
        color: var(--warn);
        background: #fff6de;
        border-color: #ead9aa;
      }

      .dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: #94a3a0;
      }

      .dot.live {
        background: #16835f;
      }

      .chat-wrap {
        width: min(880px, calc(100vw - 28px));
        margin: 0 auto;
        padding: 22px 0 180px;
      }

      .notice {
        margin: 0 0 18px;
        padding: 11px 14px;
        border: 1px solid #ead9aa;
        border-radius: 14px;
        background: #fff8e7;
        color: #6e4a10;
        line-height: 1.45;
        font-size: 0.92rem;
      }

      .examples {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 0 0 20px;
      }

      .chip {
        padding: 8px 11px;
        border-radius: 999px;
        border: 1px solid var(--line);
        color: #2f3a44;
        background: var(--surface);
        box-shadow: 0 1px 2px rgba(20, 28, 34, 0.04);
      }

      .chip:hover {
        border-color: var(--line-strong);
        background: var(--surface-soft);
      }

      .messages {
        display: grid;
        gap: 18px;
      }

      .empty {
        margin-top: 11vh;
        text-align: center;
        color: var(--muted);
      }

      .empty h1 {
        margin: 0 0 10px;
        color: var(--text);
        font-size: clamp(1.8rem, 5vw, 3rem);
        letter-spacing: -0.02em;
      }

      .empty p {
        margin: 0 auto;
        max-width: 560px;
        line-height: 1.55;
      }

      .message-row {
        display: grid;
        grid-template-columns: 36px minmax(0, 1fr);
        gap: 12px;
      }

      .message-row.user {
        grid-template-columns: minmax(0, 1fr) 36px;
      }

      .avatar {
        width: 34px;
        height: 34px;
        border-radius: 11px;
        display: grid;
        place-items: center;
        color: white;
        font-size: 0.78rem;
        font-weight: 800;
        background: var(--assistant);
      }

      .user .avatar {
        background: var(--user);
        grid-column: 2;
      }

      .bubble {
        width: fit-content;
        max-width: 100%;
        padding: 13px 15px;
        border-radius: 16px;
        border: 1px solid var(--line);
        background: var(--surface);
        box-shadow: 0 1px 3px rgba(20, 28, 34, 0.04);
      }

      .user .bubble {
        justify-self: end;
        background: #edf3f8;
        border-color: #d7e3ed;
      }

      .assistant .bubble {
        width: 100%;
      }

      .message-meta {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        margin: 0 0 8px;
      }

      .name {
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 700;
      }

      .label {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 3px 8px;
        font-size: 0.74rem;
        border: 1px solid var(--line);
        color: var(--muted);
        background: var(--surface-soft);
      }

      .label.answered {
        color: var(--accent-dark);
        background: var(--accent-soft);
        border-color: #c7e2d9;
      }

      .label.abstained,
      .label.weak {
        color: var(--warn);
        background: #fff6de;
        border-color: #ead9aa;
      }

      .label.failed {
        color: var(--danger);
        background: #fff0f0;
        border-color: #edcaca;
      }

      .content {
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        line-height: 1.62;
      }

      .typing {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        color: var(--muted);
      }

      .typing span {
        width: 6px;
        height: 6px;
        border-radius: 99px;
        background: currentColor;
        opacity: 0.35;
        animation: blink 1.2s infinite;
      }

      .typing span:nth-child(2) {
        animation-delay: 0.15s;
      }

      .typing span:nth-child(3) {
        animation-delay: 0.3s;
      }

      @keyframes blink {
        0%, 80%, 100% { opacity: 0.25; transform: translateY(0); }
        40% { opacity: 0.9; transform: translateY(-2px); }
      }

      details.sources,
      details.debug,
      details.settings {
        margin-top: 12px;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--surface-warm);
        overflow: hidden;
      }

      details.sources summary,
      details.debug summary,
      details.settings summary {
        cursor: pointer;
        list-style: none;
        padding: 10px 12px;
        font-weight: 700;
        color: #3b434b;
      }

      details.sources summary::-webkit-details-marker,
      details.debug summary::-webkit-details-marker,
      details.settings summary::-webkit-details-marker {
        display: none;
      }

      .source-list {
        display: grid;
        gap: 10px;
        padding: 0 12px 12px;
      }

      .source-card {
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 10px;
        background: var(--surface);
      }

      .source-card header {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 6px;
        color: var(--muted);
        font-size: 0.82rem;
      }

      .source-card p {
        margin: 0;
        color: #333b42;
        line-height: 1.5;
        font-size: 0.92rem;
      }

      .debug pre {
        margin: 0;
        padding: 0 12px 12px;
        max-height: 340px;
        overflow: auto;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 0.78rem;
        color: #2d343a;
      }

      .composer-shell {
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 20;
        border-top: 1px solid var(--line);
        background: rgba(247, 247, 245, 0.9);
        backdrop-filter: blur(16px);
      }

      .composer {
        width: min(880px, calc(100vw - 28px));
        margin: 0 auto;
        padding: 12px 0 16px;
      }

      .composer-box {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 10px;
        align-items: end;
        padding: 10px;
        border: 1px solid var(--line-strong);
        border-radius: 20px;
        background: var(--surface);
        box-shadow: var(--shadow);
      }

      textarea {
        width: 100%;
        min-height: 46px;
        max-height: 140px;
        resize: none;
        border: 0;
        outline: 0;
        padding: 10px 8px;
        line-height: 1.45;
        color: var(--text);
        background: transparent;
      }

      .send {
        width: 42px;
        height: 42px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        background: var(--accent);
        color: white;
        font-size: 1rem;
        font-weight: 800;
      }

      .send:disabled {
        opacity: 0.55;
        cursor: not-allowed;
      }

      .settings {
        margin-bottom: 10px;
        background: rgba(255, 255, 255, 0.78);
      }

      .settings-grid {
        padding: 0 12px 12px;
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
      }

      .field {
        display: grid;
        gap: 4px;
      }

      .field span {
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 700;
      }

      input {
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 8px 9px;
        background: var(--surface);
      }

      .toggles {
        padding: 0 12px 12px;
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
        color: var(--muted);
        font-size: 0.86rem;
      }

      .footer-note {
        margin: 8px 4px 0;
        color: var(--muted);
        font-size: 0.78rem;
      }

      @media (max-width: 720px) {
        .topbar-inner {
          min-height: 74px;
          align-items: flex-start;
          flex-direction: column;
          justify-content: center;
          padding: 10px 0;
        }

        .status-strip {
          justify-content: flex-start;
        }

        .settings-grid {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .message-row,
        .message-row.user {
          grid-template-columns: 1fr;
        }

        .avatar,
        .user .avatar {
          display: none;
        }
      }
    </style>
  </head>
  <body>
    <div class="app">
      <header class="topbar">
        <div class="topbar-inner">
          <div class="brand">
            <div class="mark" aria-hidden="true">W</div>
            <div class="brand-title">
              <strong>WebbGPT 0.2</strong>
              <span id="checkpointPath" title="__CHECKPOINT_PATH__">__CHECKPOINT_LABEL__</span>
            </div>
          </div>
          <div class="status-strip" aria-label="Server status">
            <span class="badge good"><span id="statusDot" class="dot"></span><span id="serverStatus">checking</span></span>
            <span class="badge">mode <strong id="modeBadge">__MODEL_MODE__</strong></span>
            <span class="badge">device <strong id="deviceBadge">loading</strong></span>
            <span id="ragBadge" class="badge">RAG <strong>__RAG_MODE__</strong></span>
          </div>
        </div>
      </header>

      <main class="chat-wrap">
        <p class="notice">Local-MVP research model. Outputs may drift. RAG sources appear collapsed when available, and weak generations are labeled honestly.</p>
        <div id="examples" class="examples" aria-label="Example prompts"></div>
        <section id="messages" class="messages" aria-live="polite">
          <div class="empty">
            <h1>Ask WebbGPT 0.2</h1>
            <p>Use the prompt chips or ask a question. RAG can add source context, but the local-MVP model output stays visible so the demo shows both strengths and limitations.</p>
          </div>
        </section>
      </main>

      <footer class="composer-shell">
        <form id="composer" class="composer">
          <details class="settings">
            <summary>Settings</summary>
            <div class="settings-grid">
              <label class="field">
                <span>Max tokens</span>
                <input id="maxTokensInput" type="number" min="1" max="512" step="1" value="__MAX_NEW_TOKENS__" />
              </label>
              <label class="field">
                <span>Temperature</span>
                <input id="temperatureInput" type="number" min="0" max="2" step="0.05" value="__TEMPERATURE__" />
              </label>
              <label class="field">
                <span>Top-K</span>
                <input id="topKInput" type="number" min="0" max="500" step="1" value="__TOP_K__" />
              </label>
              <label class="field">
                <span>Top-P</span>
                <input id="topPInput" type="number" min="0.01" max="1" step="0.01" value="__TOP_P__" />
              </label>
            </div>
            <div class="toggles">
              <label><input id="ragToggle" type="checkbox" checked /> Use RAG</label>
              <label><input id="sourcesToggle" type="checkbox" checked /> Show sources</label>
            </div>
          </details>
          <div class="composer-box">
            <textarea id="promptInput" rows="1" placeholder="Message WebbGPT 0.2..." aria-label="Prompt"></textarea>
            <button id="sendButton" class="send" type="submit" aria-label="Send">↑</button>
          </div>
          <p id="statusLine" class="footer-note">Enter sends. Shift+Enter adds a new line.</p>
        </form>
      </footer>
    </div>

    <script>
      const messages = [];
      const examples = __EXAMPLES_JSON__;
      const messagesEl = document.getElementById("messages");
      const examplesEl = document.getElementById("examples");
      const composer = document.getElementById("composer");
      const promptInput = document.getElementById("promptInput");
      const sendButton = document.getElementById("sendButton");
      const statusLine = document.getElementById("statusLine");
      const maxTokensInput = document.getElementById("maxTokensInput");
      const temperatureInput = document.getElementById("temperatureInput");
      const topKInput = document.getElementById("topKInput");
      const topPInput = document.getElementById("topPInput");
      const ragToggle = document.getElementById("ragToggle");
      const sourcesToggle = document.getElementById("sourcesToggle");

      function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
      }

      function normalizeLabel(status) {
        if (!status) return "Generated";
        if (status.generation_failed && !status.degenerate_output) return "Generation failed";
        if (status.degenerate_output) return "Weak generation";
        if (status.abstained) return "Abstained";
        return status.final_label || "Generated";
      }

      function labelClass(label) {
        if (label === "Generated" || label === "Generated with sources") return "answered";
        if (label === "Abstained") return "abstained";
        if (label === "Weak generation") return "weak";
        return "failed";
      }

      function requestPayload(prompt) {
        const topKValue = topKInput.value.trim() === "" ? null : Number.parseInt(topKInput.value, 10);
        return {
          prompt,
          tools: ragToggle.checked,
          citations: true,
          max_new_tokens: Number.parseInt(maxTokensInput.value, 10),
          temperature: Number.parseFloat(temperatureInput.value),
          top_k: Number.isFinite(topKValue) ? topKValue : null,
          top_p: Number.parseFloat(topPInput.value),
        };
      }

      function scrollLatest() {
        window.requestAnimationFrame(() => window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" }));
      }

      function renderSources(metadata) {
        if (!sourcesToggle.checked) {
          return document.createDocumentFragment();
        }
        const rag = metadata?.rag || {};
        const hits = rag.hits || [];
        const wrapper = document.createDocumentFragment();
        if (!hits.length) {
          if (metadata?.status?.abstained) {
            const note = el("div", "footer-note", "No reliable source found.");
            wrapper.appendChild(note);
          }
          return wrapper;
        }
        const details = el("details", "sources");
        details.open = false;
        details.appendChild(el("summary", "", `Sources available (${hits.length})`));
        const list = el("div", "source-list");
        for (const hit of hits) {
          const card = el("article", "source-card");
          const header = document.createElement("header");
          header.appendChild(el("span", "", hit.chunk_id || "chunk"));
          header.appendChild(el("span", "", hit.score === undefined ? "score n/a" : `score ${hit.score}`));
          card.appendChild(header);
          card.appendChild(el("p", "", hit.source_file || "source unavailable"));
          const meta = el("p", "", `risk: ${hit.risk_level || "n/a"} · use: ${hit.allowed_use || "n/a"}`);
          card.appendChild(meta);
          card.appendChild(el("p", "", hit.text_preview || ""));
          list.appendChild(card);
        }
        details.appendChild(list);
        wrapper.appendChild(details);
        return wrapper;
      }

      function renderDebug(metadata) {
        const details = el("details", "debug");
        details.appendChild(el("summary", "", "Run details"));
        const pre = document.createElement("pre");
        pre.textContent = JSON.stringify(metadata || {}, null, 2);
        details.appendChild(pre);
        return details;
      }

      function render() {
        messagesEl.innerHTML = "";
        if (!messages.length) {
          const empty = el("div", "empty");
          empty.appendChild(el("h1", "", "Ask WebbGPT 0.2"));
          empty.appendChild(el("p", "", "Use the prompt chips or ask a question. RAG can add source context, but the local-MVP model output stays visible so the demo shows both strengths and limitations."));
          messagesEl.appendChild(empty);
          return;
        }
        for (const message of messages) {
          const row = el("article", `message-row ${message.role}`);
          const avatar = el("div", "avatar", message.role === "user" ? "You" : "W");
          const bubble = el("div", "bubble");
          if (message.role === "assistant") {
            const metadata = message.metadata || {};
            const label = message.pending ? "Thinking" : (message.streaming ? "Streaming" : normalizeLabel(metadata.status));
            const meta = el("div", "message-meta");
            meta.appendChild(el("span", "name", "WebbGPT 0.2"));
            meta.appendChild(el("span", `label ${message.pending || message.streaming ? "" : labelClass(label)}`, label));
            if (metadata.rag?.retrieved_hits) {
              meta.appendChild(el("span", "label answered", `${metadata.rag.retrieved_hits} source${metadata.rag.retrieved_hits === 1 ? "" : "s"}`));
            }
            bubble.appendChild(meta);
          }
          const content = el("div", "content", message.content);
          if (message.pending && !message.content) {
            const typing = el("div", "typing");
            typing.appendChild(el("span"));
            typing.appendChild(el("span"));
            typing.appendChild(el("span"));
            content.appendChild(typing);
          }
          bubble.appendChild(content);
          if (message.role === "assistant" && !message.pending) {
            bubble.appendChild(renderSources(message.metadata || {}));
            bubble.appendChild(renderDebug(message.metadata || {}));
          }
          if (message.role === "user") {
            row.appendChild(bubble);
            row.appendChild(avatar);
          } else {
            row.appendChild(avatar);
            row.appendChild(bubble);
          }
          messagesEl.appendChild(row);
        }
        scrollLatest();
      }

      function resizeComposer() {
        promptInput.style.height = "auto";
        promptInput.style.height = Math.min(promptInput.scrollHeight, 140) + "px";
      }

      function parseSseEvent(raw) {
        let event = "message";
        const dataLines = [];
        for (const line of raw.split("\\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
        }
        return { event, data: dataLines.join("\\n") };
      }

      async function streamPrompt(prompt, assistant) {
        const response = await fetch("/generate_stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestPayload(prompt)),
        });
        if (!response.ok || !response.body) {
          throw new Error(await response.text() || `Streaming failed with ${response.status}`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let finalMetadata = {};
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split("\\n\\n");
          buffer = parts.pop() || "";
          for (const part of parts) {
            if (!part.trim()) continue;
            const parsed = parseSseEvent(part);
            const payload = parsed.data ? JSON.parse(parsed.data) : {};
            if (parsed.event === "delta") {
              assistant.pending = false;
              assistant.streaming = true;
              assistant.content += payload.text || "";
              render();
            } else if (parsed.event === "metadata") {
              finalMetadata = payload.metadata || {};
            } else if (parsed.event === "error") {
              throw new Error(payload.message || "Generation failed");
            }
          }
        }
        assistant.pending = false;
        assistant.streaming = false;
        assistant.metadata = finalMetadata;
        render();
      }

      async function sendPrompt() {
        const prompt = promptInput.value.trim();
        if (!prompt || sendButton.disabled) return;
        messages.push({ role: "user", content: prompt });
        const assistant = { role: "assistant", content: "", pending: true, metadata: {} };
        messages.push(assistant);
        promptInput.value = "";
        resizeComposer();
        render();
        sendButton.disabled = true;
        statusLine.textContent = "Generating...";
        try {
          await streamPrompt(prompt, assistant);
          statusLine.textContent = "Response complete.";
        } catch (error) {
          assistant.pending = false;
          assistant.content = error.message || "Request failed.";
          assistant.metadata = { status: { final_label: "Generation failed", generation_failed: true, degenerate_output: false } };
          render();
          statusLine.textContent = "Request failed.";
        } finally {
          sendButton.disabled = false;
          promptInput.focus();
        }
      }

      composer.addEventListener("submit", (event) => {
        event.preventDefault();
        sendPrompt();
      });

      promptInput.addEventListener("input", resizeComposer);
      promptInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          sendPrompt();
        }
      });

      for (const prompt of examples) {
        const chip = el("button", "chip", prompt);
        chip.type = "button";
        chip.addEventListener("click", () => {
          promptInput.value = prompt;
          resizeComposer();
          promptInput.focus();
        });
        examplesEl.appendChild(chip);
      }

      fetch("/status")
        .then((response) => response.json())
        .then((payload) => {
          document.getElementById("serverStatus").textContent = payload.status || "ok";
          document.getElementById("statusDot").classList.add("live");
          document.getElementById("modeBadge").textContent = payload.model_mode || "__MODEL_MODE__";
          document.getElementById("deviceBadge").textContent = payload.device || "unknown";
          const rag = payload.rag && payload.rag.enabled ? "on" : "off";
          document.getElementById("ragBadge").innerHTML = `RAG <strong>${rag}</strong>`;
          const path = payload.checkpoint_path || "__CHECKPOINT_PATH__";
          document.getElementById("checkpointPath").textContent = path.split("/").slice(-2).join("/");
          document.getElementById("checkpointPath").title = path;
        })
        .catch(() => {
          document.getElementById("serverStatus").textContent = "offline";
        });

      render();
      resizeComposer();
      promptInput.focus();
    </script>
  </body>
</html>
"""
    for key, value in replacements.items():
        html = html.replace(key, value)
    return html
