/**
 * VANGUARD Mars assistant — right-side chat panel → POST /agent/chat
 */

const STORAGE_MINIMIZED = "vanguard-agent-chat-minimized";
const TYPING_INDICATOR_ID = "agent-chat-typing";

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** API may return reply as string; coerce objects so we never render [object Object]. */
function replyToString(reply) {
  if (reply == null) return "";
  if (typeof reply === "string") return reply;
  if (typeof reply === "number" || typeof reply === "boolean") return String(reply);
  if (typeof reply === "object") {
    const o = /** @type {Record<string, unknown>} */ (reply);
    for (const key of ["text", "content", "message", "reply", "response"]) {
      const v = o[key];
      if (typeof v === "string" && v.trim()) return v;
    }
    try {
      return JSON.stringify(reply, null, 2);
    } catch {
      return String(reply);
    }
  }
  return String(reply);
}

function stripHtmlTags(text) {
  return String(text)
    .replace(/<br\s*\/?>/gi, " ")
    .replace(/<[^>]+>/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * @param {{ reply?: unknown, structured?: { summary?: string, best_site?: Record<string, unknown> } }} data
 */
function assistantSummaryFromResponse(data) {
  const structured = data.structured;
  if (structured && typeof structured.summary === "string" && structured.summary.trim()) {
    return stripHtmlTags(structured.summary);
  }
  return stripHtmlTags(replyToString(data.reply));
}

/** One-line facts from Pydantic structured output (coordinates, score, contributions). */
function formatStructuredDetails(structured) {
  if (!structured || typeof structured !== "object") return "";
  const site = /** @type {Record<string, unknown>} */ (structured).best_site;
  if (!site || typeof site !== "object") return "";
  const lat = site.latitude;
  const lon = site.longitude;
  const score = site.landing_score_percent;
  const interp = site.interpretation;
  if (!Number.isFinite(Number(lat)) || !Number.isFinite(Number(lon))) return "";
  let line = `Location: ${lat}°N, ${lon}°E · Score: ${score}% (${interp || "—"})`;
  const contribs = site.top_contributions;
  if (Array.isArray(contribs) && contribs.length) {
    const parts = contribs.map((c) => {
      if (!c || typeof c !== "object") return "";
      const o = /** @type {{ property_name?: string, contribution_percent?: number }} */ (c);
      return `${o.property_name || "?"} ${o.contribution_percent ?? "?"}%`;
    });
    line += ` · Top: ${parts.filter(Boolean).join(", ")}`;
  }
  return line;
}

/**
 * @param {{
 *   getApiBase: () => string,
 *   getCoordinateContext: () => { lat?: number, lon?: number, landingScore?: number } | null,
 *   getScoringWeights?: () => Record<string, number> | null,
 *   executeUiActions?: (actions: unknown[]) => Promise<void> | void,
 * }} options
 */
export function initAgentChat(options) {
  const shell = document.getElementById("agent-chat");
  const toggleBtn = document.getElementById("agentChatToggle");
  const minimizeBtn = document.getElementById("agentChatMinimize");
  const messagesEl = document.getElementById("agentChatMessages");
  const form = document.getElementById("agentChatForm");
  const input = document.getElementById("agentChatInput");
  const sendBtn = document.getElementById("agentChatSend");
  const statusEl = document.getElementById("agentChatStatus");

  if (!shell || !messagesEl || !form || !input) {
    return;
  }

  /** @type {{ role: 'user' | 'assistant' | 'system', text: string, details?: string }[]} */
  const transcript = [];

  function setMinimized(min) {
    shell.classList.toggle("agent-chat--minimized", min);
    if (toggleBtn) {
      toggleBtn.setAttribute("aria-expanded", min ? "false" : "true");
    }
    try {
      sessionStorage.setItem(STORAGE_MINIMIZED, min ? "1" : "0");
    } catch {
      /* ignore */
    }
    if (!min) {
      input.focus();
    }
  }

  setMinimized(sessionStorage.getItem(STORAGE_MINIMIZED) === "1");

  toggleBtn?.addEventListener("click", () => {
    setMinimized(!shell.classList.contains("agent-chat--minimized"));
  });
  minimizeBtn?.addEventListener("click", () => {
    setMinimized(true);
  });

  function showTypingIndicator() {
    hideTypingIndicator();
    const wrap = document.createElement("div");
    wrap.id = TYPING_INDICATOR_ID;
    wrap.className = "agent-chat-msg agent-chat-msg--assistant agent-chat-msg--typing";
    wrap.setAttribute("aria-live", "polite");
    wrap.innerHTML = `
      <div class="agent-chat-msg__bubble agent-chat-msg__bubble--typing">
        <div class="agent-chat-typing" role="status" aria-label="Assistant is responding">
          <span class="agent-chat-typing__dot" aria-hidden="true"></span>
          <span class="agent-chat-typing__dot" aria-hidden="true"></span>
          <span class="agent-chat-typing__dot" aria-hidden="true"></span>
        </div>
      </div>`;
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function hideTypingIndicator() {
    document.getElementById(TYPING_INDICATOR_ID)?.remove();
  }

  function render() {
    hideTypingIndicator();
    if (transcript.length === 0) {
      messagesEl.innerHTML =
        '<p class="agent-chat-welcome">Ask about landing suitability or say <strong>find a good landing region</strong> for a single best site (plain-text + JSON). Named sites: <strong>show me Gale crater</strong>. Uses your scoring weights from the left panel.</p>';
      return;
    }
    messagesEl.innerHTML = "";
    for (const m of transcript) {
      const wrap = document.createElement("div");
      wrap.className = `agent-chat-msg agent-chat-msg--${m.role}`;
      const bubble = document.createElement("div");
      bubble.className = "agent-chat-msg__bubble";
      bubble.textContent = m.text;
      if (m.role === "assistant" && m.details) {
        const meta = document.createElement("div");
        meta.className = "agent-chat-msg__meta";
        meta.textContent = m.details;
        bubble.appendChild(meta);
      }
      wrap.appendChild(bubble);
      messagesEl.appendChild(wrap);
    }
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function setStatus(text, isError = false) {
    if (!statusEl) return;
    statusEl.textContent = typeof text === "string" ? text : replyToString(text);
    statusEl.classList.toggle("agent-chat-status--error", isError);
  }

  function buildPayload(userText) {
    let message = userText.trim();
    const ctx = options.getCoordinateContext?.();
    if (ctx && Number.isFinite(ctx.lat) && Number.isFinite(ctx.lon)) {
      let extra = `\n\n[App context: user is focused on Mars at lat ${ctx.lat}°N, lon ${ctx.lon}°E`;
      if (Number.isFinite(ctx.landingScore)) {
        extra += `; their last landing suitability score on the globe was ${ctx.landingScore}%`;
      }
      extra +=
        ". If they ask about 'here' or this location, call focus_mars_coordinates or analyze_landing_site. If they ask to show a named site, call focus_mars_site.]";
      message += extra;
    }
    const payload = { message };
    const weights = options.getScoringWeights?.();
    if (weights && typeof weights === "object") {
      payload.scoring_weights = weights;
    }
    return payload;
  }

  let inFlight = false;

  async function sendMessage() {
    const text = input.value.trim();
    if (!text || inFlight) return;

    const base = options.getApiBase?.() ?? "";
    if (!base) {
      setStatus(
        "Serve this page from Flask or set vanguard-api-base so /agent/chat is reachable.",
        true
      );
      return;
    }

    transcript.push({ role: "user", text });
    input.value = "";
    render();
    showTypingIndicator();
    setMinimized(false);

    inFlight = true;
    if (sendBtn) sendBtn.disabled = true;
    input.disabled = true;
    setStatus("");

    try {
      const res = await fetch(`${base}/agent/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPayload(text)),
      });
      const data = await res.json().catch(() => ({}));
      hideTypingIndicator();
      if (!res.ok || !data.success) {
        const err =
          replyToString(data.error) ||
          (res.status === 503
            ? "Assistant unavailable (check OPENAI_API_KEY in server .env)."
            : `Request failed (${res.status})`);
        transcript.push({ role: "system", text: err });
        setStatus(err, true);
      } else {
        const summary = assistantSummaryFromResponse(data) || "(empty reply)";
        const details = formatStructuredDetails(data.structured);
        transcript.push({
          role: "assistant",
          text: summary,
          ...(details ? { details } : {}),
        });
        const actions = data.ui_actions;
        if (Array.isArray(actions) && actions.length > 0 && options.executeUiActions) {
          showTypingIndicator();
          setStatus("Updating globe…");
          await options.executeUiActions(actions);
          hideTypingIndicator();
          setStatus("");
        } else {
          setStatus("");
        }
      }
    } catch (e) {
      hideTypingIndicator();
      const err = `Could not reach assistant: ${e.message}`;
      transcript.push({ role: "system", text: err });
      setStatus(err, true);
    } finally {
      inFlight = false;
      hideTypingIndicator();
      if (sendBtn) sendBtn.disabled = false;
      input.disabled = false;
      render();
      input.focus();
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    void sendMessage();
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendMessage();
    }
  });
}
