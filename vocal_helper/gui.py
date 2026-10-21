"""
Vocal Helper — transcript-viewer GUI ("who spoke when" bench).

This module holds nothing but the self-contained HTML document served by the
FastAPI app at ``GET /gui`` (see :mod:`vocal_helper.api`). It is deliberately
build-step-free: one string of HTML + Tailwind (via CDN) + vanilla ES-module
JavaScript. There is no bundler, no framework, no npm — the whole page is a
static asset the API returns verbatim.

Where the words live
--------------------
In the markup, in English. Each one is wrapped as ``{{_("…")}}`` in the HTML,
or listed in :data:`_JS_NAME_TO_SOURCE` for the script, and
:class:`os_helper.Catalog` translates it through the compiled ``gettext``
catalogues under ``locales/`` (``CODING.md`` 21.3).

English is the msgid, so it needs no catalogue and cannot fall out of sync with
one; a string the French catalogue has not caught up with renders as readable
English instead of as a key. The cost is the mirror image: rewording the
English here orphans its French entry, and gettext would quietly serve the
English. ``tests/test_gui_i18n.py`` exists to catch that — it hands every
source string to :meth:`os_helper.Catalog.problems`, so an orphaned
translation is a red build rather than a half-reverted page.

:func:`render_gui` resolves a language and returns the finished page.

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
Warith Harchaoui, Ph.D. , https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import functools
import html
import json
import re
from pathlib import Path

import os_helper as osh

# Inside the package, not beside it: a file one level up is not package data
# and does not travel in the wheel. best-engine shipped exactly that bug —
# `pip install` gave a package whose every GUI string raised FileNotFoundError.
_LOCALES_DIR = Path(__file__).resolve().parent / "locales"

#: The gettext domain, and therefore each ``.mo`` file's stem.
_DOMAIN = "vocal_helper"

#: The language served when the request names one we cannot render.
_DEFAULT_LANGUAGE = "fr"


@functools.lru_cache(maxsize=1)
def _catalog() -> osh.Catalog:
    """Resolve the catalogue directory once, through the suite's shared loader."""
    return osh.Catalog.from_package(_LOCALES_DIR, _DOMAIN, default_language=_DEFAULT_LANGUAGE)


def supported_languages() -> tuple[str, ...]:
    """Language codes the GUI can render: English plus every compiled catalogue."""
    return _catalog().languages


def default_language() -> str:
    """The language an unknown or absent ``?lang=`` falls back to."""
    return _catalog().default_language


# The language toggle is computed, not translated. Its three parts all mean
# "the other language", which is data rather than copy: a msgid of
# ``/gui?lang=fr`` whose translation is ``/gui?lang=en`` would be nonsense, and
# `Catalog.problems` would rightly treat it as prose that had drifted.
#
#: Flag shown for a language — the flag of that language, not of the page.
_FLAGS: dict[str, str] = {"fr": "🇫🇷", "en": "🇬🇧"}

#: How each language names itself. Deliberately NOT translated: a French page
#: offering English says "English version" in English, which is what a reader
#: scanning for their own language actually looks for.
_ENDONYMS: dict[str, str] = {"fr": "Version française", "en": "English version"}


# The inline script's strings. The JS names are the contract between this
# module and the script below; the English beside each one is the msgid, so
# the script's copy lives in the same place as the markup's and is checked
# by the same test. Injected as JSON, which is the one encoding an
# apostrophe cannot break out of — and French copy is full of them.
_JS_NAME_TO_SOURCE: dict[str, str] = {
    "needSource": "Pick a file or paste a URL first.",
    "transcribing": "Transcribing locally…",
    "errorPrefix": "Error ",
    "noSpeech": "(no speech detected — empty or silent audio)",
    "done": "Done — {count} utterance(s), processed locally.",
    "requestFailed": "Request failed: {error}. Is the local server running?",
}


#: One ``{{_("…")}}`` site in the template. Non-greedy and newline-free, so a
#: run of them on one line stays separate and an unterminated one is left
#: alone rather than swallowing the rest of the page.
_SOURCE_RE = re.compile(r'\{\{_\("(.*?)"\)\}\}')

_GUI_TEMPLATE: str = r"""<!doctype html>
<html lang="{{LANG}}" class="h-full">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{_("Vocal Helper — Transcript Viewer")}}</title>
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
      <!-- Language toggle: a flag that links to the other locale of this page. -->
      <a href="{{LANG_HREF}}" aria-label="{{LANG_ARIA}}"
         class="float-right text-xl hover:opacity-70">{{LANG_FLAG}}</a>
      <h1 class="text-2xl font-semibold tracking-tight">{{_("Vocal Helper — Transcript Viewer")}}</h1>
      <p class="mt-1 text-sm text-slate-600 dark:text-slate-400">
        {{_("Drop an audio file or paste a URL, run diarized transcription locally, then read a speaker-labelled, colour-coded transcript and rolling summary.")}}
      </p>
      <p class="mt-2 inline-block rounded-full border border-slate-300 px-3 py-0.5 text-xs
                text-slate-600 dark:border-slate-600 dark:text-slate-400">
        {{_("Runs locally — your audio never leaves this machine")}}
      </p>
    </header>

    <!-- 1) Source: a local file (kept client-side) OR a URL (fetched by the
         LOCAL server, not the browser). One of the two is required. -->
    <section class="mb-5">
      <label for="file" class="block text-sm font-medium mb-1">{{_("Audio source")}}</label>
      <div id="drop" tabindex="0" role="button" aria-label="{{_("Choose an audio file")}}"
           class="flex flex-col items-center justify-center rounded-xl border-2 border-dashed
                  border-slate-300 bg-white px-4 py-8 text-center cursor-pointer
                  focus:outline-none focus:ring-2 focus:ring-blue-500 hover:border-blue-400
                  dark:border-slate-600 dark:bg-slate-800">
        <p class="text-sm text-slate-500 dark:text-slate-400">{{_("Drop a file here, or click to choose")}}</p>
        <p id="filename" class="mt-2 text-sm font-medium text-slate-800 dark:text-slate-200"></p>
        <input id="file" type="file" accept="audio/*,video/*" class="hidden" />
      </div>
      <div class="mt-3">
        <label for="url" class="block text-xs font-medium mb-1">{{_("…or paste a URL")}}
          <span class="font-normal text-slate-500">{{_("(YouTube / podcast RSS / direct media — needs the [stream] extra)")}}</span>
        </label>
        <input id="url" type="url" inputmode="url" spellcheck="false" placeholder="https://…"
               class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm
                      dark:border-slate-600 dark:bg-slate-800" />
      </div>
    </section>

    <!-- 2) Options: language, diarization backend, optional rolling summary. -->
    <section class="mb-5 grid grid-cols-2 gap-3">
      <div>
        <label for="language" class="block text-xs font-medium mb-1">{{_("language")}}
          <span class="font-normal text-slate-500">{{_("('auto' detects it)")}}</span>
        </label>
        <input id="language" value="auto" spellcheck="false"
               class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm
                      dark:border-slate-600 dark:bg-slate-800" />
      </div>
      <div>
        <label for="diar_backend" class="block text-xs font-medium mb-1">{{_("diarization backend")}}</label>
        <select id="diar_backend"
                class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm
                       dark:border-slate-600 dark:bg-slate-800">
          <option value="auto">{{_("auto — router picks by length")}}</option>
          <option value="pyannote">{{_("pyannote (robust, long)")}}</option>
          <option value="nemo">{{_("nemo (short ≤4-speaker)")}}</option>
          <option value="sherpa">{{_("sherpa (torch-free ONNX)")}}</option>
        </select>
      </div>
      <div class="col-span-2 flex items-center gap-2">
        <input id="llm" type="checkbox" class="h-4 w-4" />
        <label for="llm" class="text-sm">{{_("Add a local LLM rolling summary")}}</label>
      </div>
    </section>

    <!-- 3) Run button + status line. -->
    <section class="mb-6">
      <button id="run"
              class="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white
                     hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500
                     disabled:opacity-50">
        {{_("Transcribe locally")}}
      </button>
      <span id="status" class="ml-3 text-sm text-slate-600 dark:text-slate-400"
            role="status" aria-live="polite"></span>
    </section>

    <!-- 4) Rolling summary panel (only shown when the LLM stage ran). -->
    <section id="summary-panel" class="mb-6 hidden rounded-xl border border-amber-200 bg-amber-50 p-4
                                       dark:border-amber-900 dark:bg-amber-950/40">
      <h2 class="mb-2 text-sm font-semibold">{{_("Rolling summary")}}</h2>
      <p id="summary" class="text-sm whitespace-pre-wrap"></p>
    </section>

    <!-- 5) Speaker legend + colour-coded transcript. -->
    <section>
      <div id="legend" class="mb-3 flex flex-wrap gap-2" aria-label="{{_("Speakers")}}"></div>
      <div id="transcript" class="space-y-1" aria-live="polite"></div>
    </section>
  </div>

  <script type="module">
    // Every human-readable string the script writes comes from the locale
    // file, injected as JSON — the one encoding an apostrophe cannot break
    // out of, and French copy is full of them.
    const T = {{T_JSON}};

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
      if (!inputFile && !url) { status(T.needSource); return; }

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

      status(T.transcribing);
      $("run").disabled = true;
      try {
        const res = await fetch("/pipeline", { method: "POST", body: fd });
        if (!res.ok) {
          const txt = await res.text();
          status(T.errorPrefix + res.status + ": " + txt.slice(0, 300));
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
          $("transcript").textContent = T.noSpeech;
        }
        // Named fields, not concatenation: a translator can reorder the
        // sentence instead of inheriting the order JavaScript glued it in.
        status(T.done.replace("{count}", uttIndex));
      } catch (err) {
        status(T.requestFailed.replace("{error}", err));
      } finally {
        $("run").disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def render_gui(lang: str = "") -> str:
    """Render the single-page GUI in the requested language.

    Parameters
    ----------
    lang : str
        Two-letter language code. Anything outside the locale file's
        a compiled catalogue — including the empty string — falls back
        to ``meta.default_locale``, so an unknown ``?lang=`` never 500s.

    Returns
    -------
    str
        The complete HTML document for that language.

    Examples
    --------
    >>> "<!doctype html>" in render_gui("en")
    True
    >>> render_gui("tlh") == render_gui(default_language())
    True
    """
    cat = _catalog()
    resolved = cat.resolve(lang)
    translate = cat.translator_for(resolved).gettext

    # The other language, for the toggle. With two languages this is "the one
    # that is not current"; with more it is the next one round, which keeps the
    # flag a cycle rather than a dead end.
    languages = cat.languages
    other = languages[(languages.index(resolved) + 1) % len(languages)]

    def _render_source(match: re.Match[str]) -> str:
        """Translate one ``{{_("…")}}`` site and escape it for HTML.

        Escaping is not optional: these land in attributes (placeholder, alt,
        aria, title) as readily as in text nodes, and a French apostrophe in an
        attribute would close it early.
        """
        return html.escape(translate(match.group(1)), quote=True)

    page = _SOURCE_RE.sub(_render_source, _GUI_TEMPLATE)
    page = page.replace("{{LANG}}", html.escape(resolved, quote=True))
    page = page.replace("{{LANG_HREF}}", html.escape(f"/gui?lang={other}", quote=True))
    page = page.replace("{{LANG_FLAG}}", _FLAGS.get(other, other.upper()))
    page = page.replace("{{LANG_ARIA}}", html.escape(_ENDONYMS.get(other, other), quote=True))

    # The script's strings go in as JSON rather than as substituted text: it is
    # the encoding that cannot be broken out of by an apostrophe, and French
    # copy is full of them.
    table = {name: translate(source) for name, source in _JS_NAME_TO_SOURCE.items()}
    return page.replace("{{T_JSON}}", json.dumps(table, ensure_ascii=False))


def __getattr__(name: str) -> str:
    """Serve ``GUI_HTML`` on first access rather than building it at import.

    The module-level constant used to BE the page, and importers still reach
    for it. Assigning it eagerly would read and parse the locale file on every
    import of this module, including in callers that never serve the GUI — and
    a module-level ``__getattr__`` only fires for names that are NOT otherwise
    defined, so the name is deliberately left unbound rather than set to a
    placeholder.
    """
    if name == "GUI_HTML":
        return render_gui()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
