"""
Vocal Helper — transcript-viewer GUI ("who spoke when" bench).

This module holds nothing but the self-contained HTML document served by the
FastAPI app at ``GET /gui`` (see :mod:`vocal_helper.api`). It is deliberately
build-step-free: one string of HTML + Tailwind (via CDN) + vanilla ES-module
JavaScript. There is no bundler, no framework, no npm — the whole page is a
static asset the API returns verbatim.

Why a separate module
---------------------
Keeping the (long) HTML out of :mod:`vocal_helper.api` keeps the route
definitions readable and mirrors the AI Helpers suite convention (see
``audio_helper/gui.py``): swap the operation for vocal-helper's domain, keep
the same "one HTML string returned by a ``/gui`` route" plumbing.

What the page does
------------------
- Drop / pick a local audio file **or paste a URL** (YouTube / podcast RSS /
  direct media) — both stay local: the file never leaves the machine and the
  URL is fetched by the *local* server, not the browser.
- Run the full offline pipeline (VAD + diarization + STT + optional rolling
  Gemma summary) by POSTing to the same ``/pipeline`` endpoint the CLI and MCP
  surfaces use — the GUI adds zero new server logic.
- Render a **speaker-labelled, colour-coded transcript** — one stable colour
  per speaker (S0, S1, …) — with each utterance's start time, plus the rolling
  summary in its own panel.
- Utterances reveal progressively (motion-guarded) so a long transcript reads
  like it is streaming in, the honest testable subset of a live view.

Local-first
-----------
Sovereign by design: the page talks ONLY to the local vocal-helper server it
is served from. No CDN audio, no analytics, no third party sees your voice.

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

# The entire GUI is this one HTML string, returned as-is by the ``/gui`` route.
# Tailwind is pulled from a CDN so there is no build step; the JavaScript is a
# single inline ES module talking to the existing ``/pipeline`` endpoint.
GUI_HTML: str = r"""<!doctype html>
<html lang="en" class="h-full">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Vocal Helper — Transcript Viewer</title>
  <!-- Tailwind via CDN: keeps the page a single self-contained file, no build. -->
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    /* Respect users who ask for reduced motion (accessibility baseline). */
    @media (prefers-reduced-motion: reduce) {
      * { transition: none !important; animation: none !important; }
    }
    /* Progressive reveal: each utterance fades/slides in as it is appended,
       so a long transcript reads as if it were streaming (motion-guarded). */
    @media (prefers-reduced-motion: no-preference) {
      .utt { animation: fadein .28s ease-out both; }
      @keyframes fadein { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
    }
  </style>
</head>
<body class="h-full bg-slate-50 text-slate-900 antialiased dark:bg-slate-900 dark:text-slate-100">
  <div class="mx-auto max-w-3xl px-4 py-8">
    <header class="mb-6">
      <h1 class="text-2xl font-semibold tracking-tight">Vocal Helper — Transcript Viewer</h1>
      <p class="mt-1 text-sm text-slate-600 dark:text-slate-400">
        Drop an audio file or paste a URL, run diarized transcription locally,
        then read a speaker-labelled, colour-coded transcript and rolling summary.
      </p>
      <p class="mt-2 inline-block rounded-full border border-slate-300 px-3 py-0.5 text-xs
                text-slate-600 dark:border-slate-600 dark:text-slate-400">
        Runs locally — your audio never leaves this machine
      </p>
    </header>

    <!-- 1) Source: a local file (kept client-side) OR a URL (fetched by the
         LOCAL server, not the browser). One of the two is required. -->
    <section class="mb-5">
      <label for="file" class="block text-sm font-medium mb-1">Audio source</label>
      <div id="drop" tabindex="0" role="button" aria-label="Choose an audio file"
           class="flex flex-col items-center justify-center rounded-xl border-2 border-dashed
                  border-slate-300 bg-white px-4 py-8 text-center cursor-pointer
                  focus:outline-none focus:ring-2 focus:ring-blue-500 hover:border-blue-400
                  dark:border-slate-600 dark:bg-slate-800">
        <p class="text-sm text-slate-500 dark:text-slate-400">Drop a file here, or click to choose</p>
        <p id="filename" class="mt-2 text-sm font-medium text-slate-800 dark:text-slate-200"></p>
        <input id="file" type="file" accept="audio/*,video/*" class="hidden" />
      </div>
      <div class="mt-3">
        <label for="url" class="block text-xs font-medium mb-1">…or paste a URL
          <span class="font-normal text-slate-500">(YouTube / podcast RSS / direct media — needs the [stream] extra)</span>
        </label>
        <input id="url" type="url" inputmode="url" spellcheck="false" placeholder="https://…"
               class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm
                      dark:border-slate-600 dark:bg-slate-800" />
      </div>
    </section>

    <!-- 2) Options: language, diarization backend, optional rolling summary. -->
    <section class="mb-5 grid grid-cols-2 gap-3">
      <div>
        <label for="language" class="block text-xs font-medium mb-1">language
          <span class="font-normal text-slate-500">('auto' detects it)</span>
        </label>
        <input id="language" value="auto" spellcheck="false"
               class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm
                      dark:border-slate-600 dark:bg-slate-800" />
      </div>
      <div>
        <label for="diar_backend" class="block text-xs font-medium mb-1">diarization backend</label>
        <select id="diar_backend"
                class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm
                       dark:border-slate-600 dark:bg-slate-800">
          <option value="auto">auto — router picks by length</option>
          <option value="pyannote">pyannote (robust, long)</option>
          <option value="nemo">nemo (short ≤4-speaker)</option>
          <option value="sherpa">sherpa (torch-free ONNX)</option>
        </select>
      </div>
      <div class="col-span-2 flex items-center gap-2">
        <input id="llm" type="checkbox" class="h-4 w-4" />
        <label for="llm" class="text-sm">Add a local Gemma rolling summary</label>
      </div>
    </section>

    <!-- 3) Run button + status line. -->
    <section class="mb-6">
      <button id="run"
              class="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white
                     hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500
                     disabled:opacity-50">
        Transcribe locally
      </button>
      <span id="status" class="ml-3 text-sm text-slate-600 dark:text-slate-400"
            role="status" aria-live="polite"></span>
    </section>

    <!-- 4) Rolling summary panel (only shown when the LLM stage ran). -->
    <section id="summary-panel" class="mb-6 hidden rounded-xl border border-amber-200 bg-amber-50 p-4
                                       dark:border-amber-900 dark:bg-amber-950/40">
      <h2 class="mb-2 text-sm font-semibold">Rolling summary</h2>
      <p id="summary" class="text-sm whitespace-pre-wrap"></p>
    </section>

    <!-- 5) Speaker legend + colour-coded transcript. -->
    <section>
      <div id="legend" class="mb-3 flex flex-wrap gap-2" aria-label="Speakers"></div>
      <div id="transcript" class="space-y-1" aria-live="polite"></div>
    </section>
  </div>

  <script type="module">
    // --- tiny DOM helpers -------------------------------------------------
    const $ = (id) => document.getElementById(id);
    const status = (msg) => { $("status").textContent = msg; };

    // Currently-selected primary input file (kept client-side until Run).
    let inputFile = null;

    // --- file picker + drag-and-drop -------------------------------------
    const drop = $("drop");
    const fileInput = $("file");
    drop.addEventListener("click", () => fileInput.click());
    drop.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
    });
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("border-blue-500"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("border-blue-500"));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      drop.classList.remove("border-blue-500");
      if (e.dataTransfer.files.length) setInput(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", () => { if (fileInput.files.length) setInput(fileInput.files[0]); });
    function setInput(f) { inputFile = f; $("filename").textContent = f.name; }

    // --- speaker -> stable colour ----------------------------------------
    // One colour per speaker id, assigned in first-seen order. The palette is
    // WCAG-legible on white AND dark and stays distinct for up to 8 speakers;
    // beyond that it cycles (rare in practice for a single recording).
    const PALETTE = [
      "#2563eb", "#dc2626", "#059669", "#d97706",
      "#7c3aed", "#db2777", "#0891b2", "#65a30d",
    ];
    const speakerColor = new Map();
    function colorFor(spk) {
      if (!speakerColor.has(spk)) {
        speakerColor.set(spk, PALETTE[speakerColor.size % PALETTE.length]);
        renderLegend();
      }
      return speakerColor.get(spk);
    }
    function renderLegend() {
      const legend = $("legend");
      legend.textContent = "";
      for (const [spk, color] of speakerColor) {
        const chip = document.createElement("span");
        chip.className = "inline-flex items-center gap-1.5 rounded-full border border-slate-200 " +
                         "bg-white px-2.5 py-0.5 text-xs font-medium dark:border-slate-700 dark:bg-slate-800";
        const dot = document.createElement("span");
        dot.className = "h-2.5 w-2.5 rounded-full";
        dot.style.backgroundColor = color;
        chip.appendChild(dot);
        chip.appendChild(document.createTextNode(spk));
        legend.appendChild(chip);
      }
    }

    // --- render one diarized utterance as a coloured bubble --------------
    function appendUtterance(ev, index) {
      const color = colorFor(ev.speaker ?? "?");
      const row = document.createElement("div");
      row.className = "utt flex gap-3 rounded-lg border border-slate-200 bg-white p-3 " +
                      "dark:border-slate-700 dark:bg-slate-800";
      // Stagger the reveal a touch so a batch of utterances streams in visibly.
      row.style.animationDelay = Math.min(index * 40, 800) + "ms";
      row.style.borderLeft = "4px solid " + color;

      const meta = document.createElement("div");
      meta.className = "shrink-0 text-right";
      const spk = document.createElement("div");
      spk.className = "text-sm font-semibold";
      spk.style.color = color;
      spk.textContent = ev.speaker ?? "?";
      const t0 = document.createElement("div");
      t0.className = "text-xs text-slate-400 tabular-nums";
      t0.textContent = (typeof ev.t0 === "number" ? ev.t0.toFixed(1) : "?") + "s";
      meta.appendChild(spk);
      meta.appendChild(t0);

      const text = document.createElement("p");
      text.className = "text-sm leading-relaxed";
      // textContent, never innerHTML: XSS-safe by construction.
      text.textContent = ev.text;

      row.appendChild(meta);
      row.appendChild(text);
      $("transcript").appendChild(row);
    }

    // --- run: POST the source to /pipeline and render the transcript -----
    $("run").addEventListener("click", async () => {
      const url = $("url").value.trim();
      if (!inputFile && !url) { status("Pick a file or paste a URL first."); return; }

      // Reset the view for a fresh run.
      speakerColor.clear();
      $("legend").textContent = "";
      $("transcript").textContent = "";
      $("summary-panel").classList.add("hidden");
      $("summary").textContent = "";

      const fd = new FormData();
      // A file takes priority; otherwise the server fetches the URL locally.
      if (inputFile) fd.append("file", inputFile);
      if (!inputFile && url) fd.append("url", url);
      fd.append("language", $("language").value || "auto");
      fd.append("diar_backend", $("diar_backend").value);
      fd.append("llm", $("llm").checked ? "true" : "false");

      status("Transcribing locally…");
      $("run").disabled = true;
      try {
        const res = await fetch("/pipeline", { method: "POST", body: fd });
        if (!res.ok) {
          const txt = await res.text();
          status("Error " + res.status + ": " + txt.slice(0, 300));
          return;
        }
        const data = await res.json();
        let latestSummary = "";
        let uttIndex = 0;
        for (const ev of data.events) {
          if (typeof ev.summary === "string") {
            // A rolling-summary snapshot from the optional Gemma analyst stage.
            latestSummary = ev.summary;
          } else if (typeof ev.text === "string") {
            appendUtterance(ev, uttIndex++);
          }
        }
        if (latestSummary) {
          $("summary").textContent = latestSummary;
          $("summary-panel").classList.remove("hidden");
        }
        if (uttIndex === 0) {
          $("transcript").textContent = "(no speech detected — empty or silent audio)";
        }
        status("Done — " + uttIndex + " utterance(s), processed locally.");
      } catch (err) {
        status("Request failed: " + err + ". Is the local server running?");
      } finally {
        $("run").disabled = false;
      }
    });
  </script>
</body>
</html>
"""
