// @ts-check
/**
 * vocal-helper — minimal local web GUI logic.
 *
 * Sovereign by design: every request goes to the user-supplied LOCAL server
 * (default http://127.0.0.1:8000) via `fetch`; nothing else is contacted. The
 * page mirrors the CLI: "Transcribe" → POST /transcribe, "Full pipeline" →
 * POST /pipeline. No framework, no build step — a single ES module.
 */

// Grab the elements once; the page is static so these never change identity.
const form = /** @type {HTMLFormElement} */ (document.getElementById("job"));
const fileInput = /** @type {HTMLInputElement} */ (document.getElementById("file"));
const runBtn = /** @type {HTMLButtonElement} */ (document.getElementById("run"));
const pipelineOnly = /** @type {HTMLDivElement} */ (document.getElementById("pipeline-only"));
const statusEl = /** @type {HTMLParagraphElement} */ (document.getElementById("status"));
const resultEl = /** @type {HTMLPreElement} */ (document.getElementById("result"));

/**
 * Return the currently selected mode ("transcribe" | "pipeline").
 *
 * @returns {string} The value of the checked mode radio.
 */
function currentMode() {
  const checked = /** @type {HTMLInputElement} */ (
    form.querySelector('input[name="mode"]:checked')
  );
  return checked.value;
}

/**
 * Set the status line, colouring it by kind so success/error read at a glance.
 *
 * @param {string} message - Text to show (also announced to screen readers).
 * @param {"info"|"ok"|"error"} [kind="info"] - Semantic colour bucket.
 * @param {boolean} [busy=false] - When true, appends an animated ellipsis.
 */
function setStatus(message, kind = "info", busy = false) {
  statusEl.textContent = message;
  statusEl.dataset.kind = kind;
  // The .busy class drives the "…" affordance (motion-guarded in CSS).
  statusEl.classList.toggle("busy", busy);
}

// Enable the Run button only once a file is chosen — no point firing empty.
fileInput.addEventListener("change", () => {
  runBtn.disabled = fileInput.files == null || fileInput.files.length === 0;
});

// Show the diarization/LLM controls only in full-pipeline mode; transcribe
// has no speakers, so hiding them keeps the form honest (Hick's law).
form.addEventListener("change", (event) => {
  const target = /** @type {HTMLElement} */ (event.target);
  if (target instanceof HTMLInputElement && target.name === "mode") {
    pipelineOnly.hidden = currentMode() !== "pipeline";
  }
});

/**
 * Render a /transcribe response: the discovered language + the text.
 *
 * @param {{text: string, language: string}} data - The JSON body.
 */
function renderTranscribe(data) {
  // The language is DISCOVERED from the audio (see the toolbox's language rule),
  // so we surface it explicitly rather than assume it.
  const lang = data.language ? `[${data.language}] ` : "";
  resultEl.textContent = `${lang}${data.text || "(no speech detected)"}`;
}

/**
 * Render a /pipeline response: one line per event (utterance or summary).
 *
 * @param {{events: Array<Record<string, any>>, count: number}} data - JSON body.
 */
function renderPipeline(data) {
  // Clear the panel, then append each event as its own node so we can style
  // the speaker label without injecting raw HTML (XSS-safe by construction).
  resultEl.textContent = "";
  for (const ev of data.events) {
    const line = document.createElement("div");
    if (typeof ev.summary === "string") {
      // A rolling-summary snapshot from the optional Gemma analyst stage.
      line.textContent = `\n— summary —\n${ev.summary}\n`;
    } else if (typeof ev.text === "string") {
      // A diarized, transcribed utterance: [t0 speaker] text.
      const t0 = typeof ev.t0 === "number" ? ev.t0.toFixed(1) : "?";
      const spk = document.createElement("span");
      spk.className = "spk";
      spk.textContent = `[${t0}s ${ev.speaker ?? "?"}] `;
      line.appendChild(spk);
      line.appendChild(document.createTextNode(ev.text));
    }
    resultEl.appendChild(line);
  }
  if (data.count === 0) {
    resultEl.textContent = "(no events — empty or silent audio)";
  }
}

/**
 * Handle the form submit: build the multipart request and call the local API.
 *
 * @param {SubmitEvent} event - The form submit event.
 * @returns {Promise<void>}
 */
async function onSubmit(event) {
  event.preventDefault();
  const mode = currentMode();
  // Trim a trailing slash so `${base}/transcribe` never doubles up.
  const base = /** @type {HTMLInputElement} */ (
    document.getElementById("api")
  ).value.replace(/\/+$/, "");
  const endpoint = mode === "pipeline" ? "/pipeline" : "/transcribe";

  // FormData mirrors the endpoints' Form(...) fields exactly.
  const body = new FormData();
  body.append("file", /** @type {FileList} */ (fileInput.files)[0]);
  body.append("language", /** @type {HTMLInputElement} */ (document.getElementById("language")).value);
  body.append(
    "whisper_model",
    /** @type {HTMLInputElement} */ (document.getElementById("whisper_model")).value,
  );
  if (mode === "pipeline") {
    body.append("diar_backend", /** @type {HTMLSelectElement} */ (document.getElementById("diar_backend")).value);
    // Checkbox → the boolean Form field the endpoint expects.
    body.append("llm", /** @type {HTMLInputElement} */ (document.getElementById("llm")).checked ? "true" : "false");
  }

  runBtn.disabled = true;
  setStatus("Running locally", "info", true);
  resultEl.textContent = "";
  try {
    // The ONLY network call this page makes — to the user's local server.
    const res = await fetch(base + endpoint, { method: "POST", body });
    if (!res.ok) {
      // Surface the server's error text so failures are diagnosable.
      const detail = await res.text();
      throw new Error(`${res.status} ${res.statusText} — ${detail.slice(0, 300)}`);
    }
    const data = await res.json();
    if (mode === "pipeline") {
      renderPipeline(data);
    } else {
      renderTranscribe(data);
    }
    setStatus("Done — processed locally.", "ok");
  } catch (err) {
    // Network failure usually means the local server isn't running.
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(`Failed: ${msg}. Is the local server running?`, "error");
  } finally {
    runBtn.disabled = false;
  }
}

form.addEventListener("submit", onSubmit);
