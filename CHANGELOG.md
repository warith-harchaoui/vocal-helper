# Changelog

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org): from 1.0.0
on, **breaking behaviour and default changes land only in MAJOR releases; MINOR
adds features compatibly; PATCH is bug-fixes and docs.** The public API is the
names exported from `vocal_helper.__all__` plus the documented CLI flags.

## [3.3.1] - 2026-10-28

### Fixed

- **`--join-threshold --help` quoted 0.30; the diarizer defaults to 0.50.**
  `JOIN_THRESHOLD` moved when the default embedder became TitaNet, and the
  measurement justifying it sits directly above the constant. Five pieces of
  prose never moved with it: the CLI help text, the config-builder comment in
  `cli.py`, the module docstring and the `OnlineDiarStage` parameter docs in
  `diar.py`, and the pipeline tour in `__init__.py`. The help string is
  interpolated from the constant now rather than carrying its own copy, and
  the docstrings point at the constant instead of quoting a figure. The
  historical 0.30 rows in the calibration table stay: there it is a
  measurement, not a claim about the default.

- **`plan.reason` told users "pyannote 3.1".** `_PyannoteOfflineDiar` has
  defaulted to `community-1` since the 2026-10-19 bench, and the wrapper's own
  docstring still said it wraps `speaker-diarization-3.1`. The router names the
  *backend* now -- which checkpoint that backend loads is its business, and
  `router.py` is deliberately import-free so it cannot read the real one.

### Added

- Two guards, each proven by breaking it first:
  `test_cli.py::test_join_threshold_help_quotes_the_real_default` and
  `test_router.py::test_no_reason_pins_a_pyannote_checkpoint`.

### Changed

- **CI fetches ffmpeg from a pinned GitHub release instead of apt.** The
  `apt-get install ffmpeg` step pulled 89 packages from
  azure.archive.ubuntu.com, and a quiet mirror cost sibling members twelve
  minutes and a red run. The replacement lands in 1.9 s.

## [3.3.0] - 2026-10-25

### Changed

- **One command line, built with click.** The two CLI surfaces collapse into a
  single module, `vocal_helper/cli.py`. `vocal_helper/cli_argparse.py` and
  `vocal_helper/cli_click.py` are gone, as is the `vocal-helper-click` entry
  point. A flag naming the parsing library describes the implementation, not
  the interface, so there was nothing left for it to distinguish.

  Parity was measured before anything was deleted, by diffing the two parsers
  through their own introspection rather than by reading them: identical
  subcommands (`file` / `mic` / `transcribe` / `url`), identical long-option
  sets per subcommand, identical positionals, identical `--diar-backend`
  choices and no default drift on any shared option. No scenario existed only
  on the argparse side, so the surviving surface needed nothing ported into it;
  the tests that drove the old parser were rewritten against the click one
  (`click.Command.make_context` is the `parse_args` analogue — it runs the whole
  option stack without invoking the callback).

- **`click` moved out of the `[cli]` extra into the core dependencies, and the
  extra is deleted.** The CLI *is* the `vocal-helper` entry point now, so its
  parser cannot be optional — an undeclared click would ship a package whose
  command fails on a clean install while passing every test, because click is
  always present in a dev environment.

- `vocal_helper/cli.py` used to be a backwards-compat shim re-exporting the
  argparse module. It is now the CLI itself, and it owns the diarization-router
  choke-points (`_route_backend`, `_choose_file_diar`, `_offline_*_available`)
  that the click surface used to import *from* the argparse module — and that
  `vocal_helper/api.py` still imports. Deleting the argparse file without
  moving those across first would have broken the HTTP surface, not just the
  CLI.

- The `os-helper` cap widens from `<3` to `<4`: os-helper 3.0.0's only breaking
  change was dropping its own CLI, which nothing here calls.

### Fixed

- **reStructuredText markup leaked into the terminal.** Three command
  docstrings wrote the extras as RST literals, so `--help` printed the double
  backticks around `[mic]` and `[stream]` verbatim, and the group help
  advertised itself as the "click twin of the argparse CLI" — an architecture
  that no longer exists. This is invisible by reading the source, where the
  markup looks correct; only running the command shows it. `tests/test_cli.py`
  now renders every help screen and fails on a literal double backtick in any
  of them.

- **`-h` and `-V` were missing.** The group now carries
  `help_option_names = ["-h", "--help"]` and a `-V` / `--version` option, with
  the version passed explicitly from `vocal_helper.__version__` rather than
  read from the installed distribution metadata — in an editable checkout that
  metadata reports whatever was last pip-installed, so `-V` would have lied
  about the code actually being run.

- **`skills/vocal-helper/references/cli-reference.md` documented a
  `--llm-model` flag that does not exist** (the analyst's model is resolved per
  machine by best-engine-ai-helper from the committed brief, never a CLI
  lever), omitted `--jsonl` while telling readers to use the HTTP endpoint for
  machine-readable output, and gave the wrong utterance and summary formats.
  Found by running the command and diffing against the real `--help`.

### Migration

- `vocal-helper-click <cmd>` → `vocal-helper <cmd>`. The subcommands, flags and
  defaults are unchanged, so nothing but the program name moves.
- `pip install "vocal-helper[cli]"` now fails on an unknown extra. Drop it:
  `pip install vocal-helper` already ships the CLI. `[all]` no longer lists
  `cli` either.
- `from vocal_helper.cli_argparse import …` / `from vocal_helper.cli_click
  import …` → `from vocal_helper.cli import …`. The argparse-only names
  `build_parser` and `_build_pipeline_config` (re-exported as
  `cli._build_config`) are gone; the single config builder is
  `vocal_helper.cli._pipeline_config`, which takes the options as keyword
  arguments instead of an `argparse.Namespace`.

## [3.2.0] - 2026-10-21

### Changed

- **The GUI's strings moved from `locales/i18n.yaml` to a `gettext`
  catalogue**, the suite-wide mechanism as of os-helper 2.6.0. The English is
  now the source, written in the markup itself as `{{_("…")}}`; the French
  lives in `vocal_helper/locales/fr/LC_MESSAGES/vocal_helper.po` and is read
  through `os_helper.Catalog`.

  A string the French catalogue has not caught up with renders as **readable
  English** instead of as a key. The cost is the mirror image — rewording the
  English orphans its French entry, and gettext would quietly serve the
  English — so `tests/test_gui_i18n.py` hands every source string to
  `Catalog.problems`, which names both halves of that fault.

  The URL placeholder is written in the markup directly: it is the same example
  in every language. The language toggle is computed rather than translated —
  its three parts all mean "the other language", which is data, not copy.

- `tests/test_api.py` asked the locale file for the page's name and two of its
  affordances. It now reads the rendered `<title>` for the first and asks the
  catalogue by English source for the others.

## [3.1.1] - 2026-10-21

### Fixed

- **The declared `os-helper` floor was a lie as of 3.1.0.** It read
  `>=2.0.0`, while the GUI released one commit earlier resolves every string
  through `os_helper.Locales` — which landed in os-helper **2.4.0**. A user
  installing against os-helper 2.0.x gets an `AttributeError` out of the
  `/gui` route.

  Nothing would have caught it: pip resolves the NEWEST os-helper, never the
  floor, so neither CI nor a developer machine ever installs the version the
  declaration permits. The build was green the whole time.

  `tests/test_gui_i18n.py` now asserts the floor against the feature that
  needs it, so the next member to adopt a new os-helper API cannot forget.
  Verified by lowering the line and watching the guard fire.

## [3.1.0] - 2026-10-21

### Added

- **The transcript-viewer GUI speaks French and English.** Every string it
  shows now comes from `vocal_helper/locales/i18n.yaml` through
  `os_helper.Locales`, the loader shared across the suite (CODING.md 21.3),
  instead of being written into the markup. `GET /gui?lang=fr|en` picks the
  language; anything the file does not carry — including a mistyped code and
  the empty default — falls back to the declared default locale, so a bad
  `?lang=` returns a page rather than a 500. A flag in the header toggles
  between the two.

  The template is tokenised (`{{TOKEN}}` in HTML, a `T` object for the
  script), so one piece of markup serves both languages and a translator never
  opens a Python file. The script's strings are injected as JSON rather than
  substituted as text: it is the encoding an apostrophe cannot break out of,
  and French copy is full of them. The two sentences the script builds carry
  named fields (`{count}`, `{error}`) rather than being concatenated, so a
  translator can reorder them.

  `GUI_HTML` still resolves, via a module-level `__getattr__` that renders on
  access — importers that predate this keep working, and the locale file is
  not read by callers that never serve the GUI.

### Fixed

- **CI was red, and had been since 2026-10-20, on two undeclared
  dependencies.** Found while running the suite before touching it, not by
  reading the code.

  `tests/test_fanout.py` and `tests/test_pipeline_strict.py` are `async def`
  tests carrying `@pytest.mark.asyncio`, and `pytest-asyncio` was in no extra.
  Pytest does not *skip* an async test it has no plugin for — it **fails**
  it. Eleven red tests for one missing line.

  `jiwer` was declared nowhere in `pyproject.toml` at all, while
  `bench.wer_report` and `bench.cpwer_report` — both public — imported it
  bare. That is not only a CI problem: `pip install vocal-helper` shipped two
  documented functions that raise `ModuleNotFoundError: No module named
  'jiwer'` with nothing to tell the caller what to install. jiwer is now a
  `[bench]` extra, the import names it, and `[dev]` carries it so the
  assertions actually run instead of being quietly red.

  Both were invisible on a developer machine, where the packages happened to
  be installed. `tests/test_dev_extra_runs_the_suite.py` ties the declaration
  to the need: add an async test or a metric-backed assertion without its
  dependency and it fails for whoever added it, not in CI. Verified by
  removing the line and watching the guard fire.

- **`assets/landscape.svg` had no final newline**, which fails this
  repository's `end-of-file-fixer` pre-commit hook — and the CI job that
  mirrors those hooks, so that job was red alongside the test job. The file
  comes from the standpoint figure generator, which emits every `.svg` without
  one; vocal-helper is the only member whose CI runs that hook, which is why
  it is the only one that noticed.

- **A GUI test asserted the default language rather than the page's
  identity.** It looked for the literals `"transcript viewer"`, `"paste a
  url"` and `"rolling summary"`, which is no longer what a French default
  renders. It now asks the locale file what the page is called, so the copy
  can change without touching the test.

### Notes

- The locale file lives **inside** the package and is declared in
  `package-data`; three tests hold that, plus the fact that git tracks it.
  The suite has shipped this bug twice by two different routes:
  best-engine-ai-helper published wheels with no data at all until 1.6.1
  (data beside the package, patterns matching nothing), and youtube-helper's
  blanket `*.yaml` would have kept its catalogue out of git — therefore out
  of the sdist, therefore out of the wheel. Both are invisible locally, where
  the file sits on disk either way.

## [3.0.0] - 2026-10-20

A major release because two **defaults change**, which this project's own
versioning rule puts in MAJOR and nowhere else. Both are a correction of
something that was quietly wrong, and both are reversible by a keyword.

### Changed

- **A run that lost a stage no longer reports success.** `Pipeline.run()`
  used to log a crashed stage and return a normal, EMPTY transcript: the
  caller saw silence with no reason for it. The helper was named
  `_await_task_swallow`, and its own docstring recorded that this shape of
  failure once hid a broken pyannote API "undetected for hours" — the author
  had moved from silent to logged, and a log is still invisible to a program.

  Crashes are now recorded per stage and `run()` raises `StageCrashed` at the
  end if any happened. Draining still continues past a dead stage, so one
  failure never leaves the others un-awaited, and `stage_failures` is
  populated either way. **`strict=False` restores the previous leniency** —
  a choice somebody makes on purpose rather than one inherited by accident.

  Verified end to end on real audio, not against the suite: the exact mistake
  that used to yield "0 utterances, no error" now names the stage and the
  `TypeError`, and a correct run produces 11 utterances with zero failures.

- **The offline pyannote checkpoint defaults to `community-1` instead of
  3.1.** On AMI the two tie exactly at a 0.116 median DER, so a median alone
  says there is nothing to choose. The whole difference is in the tail: 3.1's
  worst meeting is 0.474 against community-1's 0.149, and the spread collapses
  from 0.412 to 0.086 — one meeting, ES2011d, carries all of it, where
  community-1 is 3.3x better. On three real 64-minute six-speaker panels it
  also over-splits least of everything measured: 2.7x, against 3.0x for 3.1,
  3.7x for Sortformer and 19.7x for the sherpa path. Same price in every other
  respect, so the one with no catastrophic case wins.

  `checkpoint="3.1"` selects the old weights. A bundle that predates
  community-1 degrades to 3.1 with a warning rather than failing.

### Added

- **`diart` as the online diarizer** (`DiartDiarStage`), joined to
  `OnlineDiarStage` rather than replacing it, because the two regimes have
  different winners and both are measured: online, diart reaches 6.04 %
  speaker error against 11.7 %, with a first label in 500 ms against 20 s;
  offline, community-1 wins on every axis above. The caveat the 6.04 % hides
  is kept in the docs: that call had two speakers, its easy case.

  The port's real content is `snapshot_to_delta`. diart re-emits its entire
  running annotation at every step, which is fine for a list and impossible
  for a queue — a consumer cannot un-receive what it already got. Snapshots
  are diffed against what has gone out, keyed by `(start, speaker)`: unseen
  keys are new, a key whose end time moved is a revision, forwarded because a
  segment that grew is news and counted because that count is the honest price
  of the 500 ms first label. The label mapping is per instance; at class level
  one session's speakers would leak into the next.

- **`vocal_helper.bench`** — scoring that answers the product's question, not
  only the leaderboard's. `speaker_label_error` ignores boundary slop, which
  DER charges for and no reader perceives: it asks how often the name against
  a paragraph is wrong. `der_report` keeps DER for ranking engines but
  decomposed, and with collar 0 and overlap scored — the conventional 0.25 s
  collar with overlap skipped deletes exactly the regions an interruption
  study is about. Miss, false alarm and confusion are never collapsed: they
  have different cures, and on the 2026-10-19 AMI bench the two local paths
  had near-identical DERs and opposite failures. `cpWER` scores under the best
  speaker mapping, so a transcript only wins when the words *and* the person
  are right; `term_error_rate` scores only the words an average hides.

- **`vocal_helper.live_metrics`** — offline quality had WER and DER, live had
  nothing. Latency is measured from the END of the speech, because that is
  when a listener starts waiting and measuring from the start rewards chopping
  speech into fragments. It ships with revision rate and never alone: text
  that jumps is unusable at any word error rate, and a pipeline that emits
  early and corrects often posts a flattering latency while being worse to
  watch. Per-queue wait separates a slow model from a starved consumer, which
  have opposite fixes. An item delivered before its audio was captured is
  counted as a violation, and the summary then says outright not to trust the
  rest.

- **`fanout`** — one producer, several consumers, with the backpressure made
  explicit rather than carried in a docstring. The slowest sink pacing every
  other one is the right default and a real trap, so it is a parameter: a
  stage whose output must be complete waits, a sink that only shows the
  present drops its oldest frame. `on_wait` reports how long each put blocked
  and which sink caused it. One deadlock was found by its own test — the
  sentinel was awaited unconditionally, so tear-down hung forever on a full
  lossy queue nobody drains; it now makes room for itself the way data does.

- **`make_embedder` and `resolve_sherpa_models` are public.** Three
  repositories were importing `_TitaNetEmbedder`, `_SherpaEmbedder` and
  `_resolve_sherpa_models`. A leading underscore depended on across
  repositories is not a dependency, it is a promise nobody made.

- **`StageCrashed`**, exported, so a caller can catch exactly this.

### Fixed

- **A cancelled fan-out could still lose its end-of-stream.** The normal path
  was fixed so the sentinel makes room for itself rather than blocking on a
  queue nobody drains; the cancellation handler kept the first version's
  shape — `put_nowait` with `QueueFull` swallowed — so tearing down a session
  whose sink was full dropped the marker and left that consumer waiting
  forever. Same failure, in the path where a full queue is likeliest and the
  docstring two lines above promised it could not happen. Both paths now go
  through one helper that never blocks and never drops; on a lossless sink
  during tear-down it discards a frame to do so, which costs nothing on a
  session being destroyed. Pinned by a test confirmed to fail beforehand.

### Documentation

- `LANDSCAPE.md` / `PAYSAGE.md` rebuilt from measurements rather than
  impressions: seventeen entries regenerated through standpoint, with the
  source CSV and the coordinate YAML committed next to the figure so the map
  is auditable rather than decorative. Four ratings come from our own benches.
  It also states what the map cannot say — the two axes carry 72 % of eight
  criteria — and keeps the section on when to pick something else.

- README and LISEZMOI no longer claim online cosine clustering is *the*
  diarization answer; after the 2026-10-19 benches that is a third of the
  picture. Both carry the two regimes with their numbers, in step section for
  section.

## [2.0.6] - 2026-10-19

### Fixed

- **`SileroVADStage` labelled every segment with a `t0` that did not name its
  first sample.** The lead-pad ring is sized in whole Silero windows —
  `maxlen = max(1, edge_pad_samples // 512)` — so it carries
  `floor(edge_pad_samples / 512)` of them, and at least one because the clamp
  never lets it reach zero. `t0`, however, was derived from the *requested*
  `edge_pad_ms`. The two disagreed at every setting:

  | `edge_pad_ms` | requested | ring carries | mismatch |
  |---|---|---|---|
  | 0 | 0 | 512 | **+32.0 ms** |
  | 50 | 800 | 512 | −18.0 ms |
  | 100 | 1600 | 1536 | −4.0 ms |
  | **200 (default)** | 3200 | 3072 | **−8.0 ms** |
  | 300 | 4800 | 4608 | −12.0 ms |

  At the default the PCM began 8 ms *after* the `t0` it was labelled with; at
  `edge_pad_ms=0` — the natural way to ask for no padding — it began 32 ms
  *before* it.

  The magnitude understates the consequence. `t0` is the anchor every ASR word
  offset is mapped through to recover absolute time, so the error did not stay
  in the VAD: it shifted **every timestamp in the segment**, uniformly and
  silently. Nothing downstream could notice, because the PCM and its label were
  consistent with each other — just not with the audio.

  `t0` is now derived from the lead actually buffered. Five parametrised tests
  pin the invariant, and were confirmed to fail against the previous behaviour
  at every setting before being accepted.

### Changed

- The pre-push gate no longer forces the integration tests on. They stream a
  specific third-party YouTube video through yt-dlp, ffmpeg and Whisper, and
  `tests/test_integration_youtube.py` already documented itself as "Skipped by
  default … Run with `pytest -v -m integration`"; the shared hook's `-m ""`
  overrode both that and this project's `addopts`. Two runs stalled at 0 % CPU
  on half-closed sockets, once for over four hours, and had to be killed. Run
  them deliberately with `pytest -m integration`.

## [2.0.5] - 2026-09-03

### Fixed

- **`__version__` drift**: `vocal_helper.__version__` was left at `2.0.3`
  when `pyproject.toml` moved to `2.0.4` for the previous release, with
  nothing catching it. Fixed, and added a `test_version_is_consistent`
  guard (the same one already present in ann-router / elbow-helper /
  best-engine-ai-helper) so this can't drift silently again.

## [2.0.4] - 2026-08-25

### Fixed

- **`SileroVADStage._consume_frame`**: closing a voiced run reset the lead
  pad ring to the single window that crossed `min_silence_ms`, instead of
  reseeding it from that run's own trailing silence (already captured in
  its `samples`). A new run's `t0` is computed as "voice onset minus
  `edge_pad_ms`" regardless of how much padding is actually queued, so
  when speech resumed before the pad ring refilled naturally, the emitted
  segment's `pcm` held less audio than `t1 - t0` implied (768 samples / 48
  ms short in the reproduction case) and its `t0` could land before the
  previous segment's `t1`. Fixed by reseeding the lead pad from the
  closed run's trailing windows instead of the single closing one. Added
  a regression test in `tests/test_vad.py`.
- **`WhisperStage._transcribe_blocking` / `_transcribe_chunk_blocking`**: a
  whisper.cpp failure was swallowed with no log line at all — a
  systematic failure (bad model path, out-of-memory, corrupt weights)
  looked exactly like ordinary silence in the transcript, with no way to
  tell the two apart. Now logged via `osh.warning`, matching the
  best-effort-failure convention already used in `llm.py`.

### Docs

- **README/LISEZMOI**: added an honest "battle-tested" / "éprouvé en
  production" paragraph (PyPI, green CI, semantic-versioned tags, and the
  package's position at the top of the suite's dependency graph, on
  `os-helper`, `best-engine-ai-helper`, `audio-helper`, `capture-helper`,
  and `podcast-helper`) per the suite-wide documentation pass.

## [2.0.3] - 2026-08-17

### Fixed

- **`resolve_diarization_engines`** leaked its downloaded temp `.zip` (the
  ~750MB diarization-engines bundle) on every fresh-machine bootstrap,
  extracting it but never unlinking the temp file. Fixed with
  `try/finally: os.unlink(...)`.

### Docs

- **README/LISEZMOI**: glossed "diarization"/"diarisation" (the tool's core
  function, never defined in plain language before), the "producer/consumer
  pipeline" architecture (a short worked analogy added in both languages),
  and "aiguilleur" (an unglossed French term that had been dropped into the
  English README with no explanation). Removed em-dash punctuation asides
  from README, LISEZMOI, EXAMPLES, and every module's opening docstring
  (`router.py`/`api.py` reworked line by line; the rest got a byline-only
  fix) per the suite's writing charter. Prose only, zero behavior change.

## [2.0.2] - 2026-08-15

### Fixed

- **CLI**: a library exception now prints one clean `Error: ...` line to
  stderr and exits 1, instead of a raw Python traceback, on both CLI
  twins. `vocal-helper-click`'s console-script entry point now points at
  a new `cli_click.main()` wrapper (was the bare `cli` group).

## [2.0.1] - 2026-08-13

### Fixed

- `requirements.txt` had drifted from `pyproject.toml`: `os-helper`/
  `audio-helper` pins were stale (`>=1.5.x`, predating the 2.0.0 migration),
  `best-engine-ai-helper` (core dep — `GemmaAnalystStage` is imported
  unconditionally) and `scipy` were missing entirely, and `pywhispercpp` was
  pinned below the version `pyproject.toml` now requires.

### Changed

- Added a minimal conda `environment.yaml`.
- Docs rewrite pass (WRITING.md/ECRITURE.md compliance) and install-command
  fixes.

## [2.0.0] - 2026-08-09

### Changed

- **LLM analyst + semantic-EOT now follow the suite brief→engine contract.**
  The model is no longer resolved by `best_engine_ai_helper.text_model()` at
  import time and is no longer a hard-coded tag. A committed
  `vocal_helper/llm.brief.yaml` describes the job (a low-latency meeting
  note-taker + lightweight end-of-turn classifier); `best-engine-ai-helper`
  resolves it, per machine, into a gitignored `vocal_helper/llm.engine.yaml`
  naming the backend (Ollama or vLLM) and model. `GemmaAnalystStage` and
  `SemanticEOTStage` now take an `engine=` descriptor (from the new
  `vocal_helper.resolve_engine()` / `best_engine_ai_helper.ensure`) and route
  every request through the synchronous `best_engine_ai_helper.llm.chat`,
  replacing the direct `ollama.Client().generate(...)` calls.
- **Entry points** (`cli_argparse`, `cli_click`, `api`) resolve the engine once
  and thread it into both stages.

### Removed

- **No more model tags or model env-vars.** Dropped `DEFAULT_MODEL` /
  `_default_analyst_model` / `_extract_response_text` / `_ensure_client` (llm.py)
  and `DEFAULT_EOT_MODEL` / `_default_eot_model` (eot.py), the
  `VOCAL_HELPER_LLM_MODEL` and `VOCAL_HELPER_EOT_MODEL` overrides, and the
  `import ollama` client path in both stages.
- **CLI/API flags** `--llm-model`, `--eot-model`, and `--ollama-host` are gone;
  the model comes only from the resolved engine. A new `--endpoint` (CLI) /
  `endpoint` (API form field) overrides the serving base URL, passed through to
  `best_engine_ai_helper.ensure(..., endpoint=…)`.
- **`ollama` dropped from the `[llm]` extra** (which now just re-lists the core
  `best-engine-ai-helper`); requests go over HTTP via `llm.chat`, so no client
  library is needed. A new **`[studies]` extra** carries `ollama` for the
  benchmark scripts under `studies/` that call the Ollama/vLLM clients directly.

### Added

- `vocal_helper.resolve_engine(endpoint=None)` — resolves (and caches, per
  machine) the LLM engine descriptor from the package's `llm.brief.yaml`.
- `vocal_helper/llm.brief.yaml` (committed); `vocal_helper/llm.engine.yaml`
  gitignored.

## [1.0.0] - 2026-08-02

First stable release. The async VAD → diarization → STT → optional-analyst
pipeline and its `vocal_helper.__all__` API have been stable across the 0.x
line; 1.0.0 commits to them on the 2.x suite foundation.

### Changed

- **Adopts the 2.x / 1.x suite majors:** `os-helper>=2.0.0,<3` (was `>=1.8.0`),
  `audio-helper>=2.0.0,<3` (was `>=1.5.5`), `best-engine-ai-helper>=1.0.0,<2`
  (was `>=0.3.0,<1` — the old `<1` cap locked out best-engine's first stable
  release), and the optional extras `capture-helper>=1.0.0,<2` (mic) and
  `podcast-helper>=1.0.0,<2` (stream).
- Development status promoted to Production/Stable.
- **CI adopts `ruff format` as a real gate.** The format check dropped its
  `continue-on-error: true` (it was informational) and now blocks like the rest
  of the suite; the test matrix is trimmed to a single Python (the full sweep
  runs locally before push).

### Fixed

- README / LISEZMOI install commands no longer self-pin to a git tag (`@v0.5.2`);
  they use `pip install vocal-helper`, which always resolves to the latest
  published release.

### Added

- `tests/test_readme_install_pin.py` guards against the stale git self-pin ever
  returning to any Markdown file.

## [0.9.0] - 2026-08-02

### Added

- **Diarization / speaker-embedding models are now fetched from the shared
  ai-helpers mirror** via the new `vocal_helper.models` registry
  (`ensure_model(name, *, allow_noncommercial=False)`, mirroring
  `video_helper.faces.models`). The registry lists the ONNX weights the
  torch-free `sherpa` path uses — `sherpa-titanet-large`
  (`nemo_en_titanet_large.onnx`), `sherpa-titanet-small`, `community1-segmentation`
  and `pyannote-segmentation-3` — and resolves each from
  `AI_HELPERS_MODEL_BASE_URL` (default
  `https://harchaoui.org/warith/ai-helpers/models/`), caching under the shared
  `~/.cache/ai-helpers/models` (override: `VOCAL_HELPER_MODEL_DIR`). No
  HuggingFace, no token.

### Changed

- **`sherpa` diarization no longer requires the ~750 MB diarization-engines
  bundle.** `_SherpaEmbedder` fetches TitaNet-large through
  `models.ensure_model("sherpa-titanet-large")` when no explicit `model_path`
  is given, and `_resolve_sherpa_models` gained a third resolution tier — the
  mirror — after the `$VH_SHERPA_*` env overrides and the bundle. Both
  overrides and the bundle keep working unchanged and still win; the mirror is
  a pure fallback, so this is fully backward compatible. `ensure_model` returns
  `None` on any failure, so a missing mirror still surfaces the existing clear
  error rather than crashing.

## [0.8.0] - 2026-08-01

### Changed

- **Model choice is delegated to the suite's model picker,
  `best-engine-ai-helper`.** The analyst stage (`GemmaAnalystStage`, `llm.py`)
  and the semantic end-of-turn classifier (`eot.py`) now read
  `best_engine_ai_helper.text_model()` for their default model tag, instead of
  the now-removed `os_helper.llm_model()`. `--llm-model` / `llm_model=` CLI
  and API defaults (argparse, click, FastAPI) follow the same resolver.
  `VOCAL_HELPER_LLM_MODEL` (analyst) and `VOCAL_HELPER_EOT_MODEL` (EOT)
  override it explicitly, same precedence as before.
- **Dependencies.** Added `best-engine-ai-helper>=0.3.0,<1` as a core
  dependency (`GemmaAnalystStage` is imported unconditionally by
  `vocal_helper.__init__`, so this mirrors how `os-helper` was core). Light
  deps (click/psutil/pyyaml/requests).
- Breaking default: the analyst/EOT default model tag is no longer the fixed
  suite constant (`qwen2.5vl:7b`) — it now tracks whatever
  `best-engine-ai-helper` selects for the current machine (or its own safe
  default). Pin `VOCAL_HELPER_LLM_MODEL`/`VOCAL_HELPER_EOT_MODEL` if you
  depend on a specific tag.

## [0.7.1] - 2026-07-23

### Added

- **Router: a known-speaker-count prior.** `select_diarization` takes a new optional
  `num_speakers` (the exact count, distinct from the `max_speakers` ceiling). When the
  router picks `sherpa`, the returned `BackendPlan` now carries `sherpa_num_clusters`
  so the consumer can pin sherpa's clustering to that count, the lever that collapses
  sherpa's 2-party telephony over-segmentation (pdbms diar-study §12.1). Leaving
  `num_speakers` unset keeps sherpa on auto-clustering, so meeting-audio behaviour is
  unchanged. Additive and backward-compatible.
- `BackendPlan.rationale`, a read-only alias for `reason` (some consumers log the
  former; it was previously always empty).

## [0.7.0] - 2026-07-23

### Added

- **`OfflineDiarStage` now exposes sherpa's clustering knobs.** Two new
  backward-compatible parameters, `sherpa_cluster_threshold` (default `0.5`) and
  `sherpa_num_clusters` (default `-1`, auto), are forwarded into sherpa-onnx's
  `FastClusteringConfig`. They were previously hardcoded, which made
  `stitch_threshold` a silent no-op on the sherpa backend and left its clustering
  untunable.

### Fixed

- **Sherpa over-segmentation on noisy telephony is now controllable.** The
  hardcoded threshold `0.5` (tuned on clean AMI meeting audio) over-segments real,
  noisy, Personally Identifiable Information (PII)-redacted phone calls into dozens of speakers (we measured ~36 on
  2-party calls). A 2026-07-23 sweep against a pyannoteAI silver ground truth showed
  that raising `sherpa_cluster_threshold` helps only slowly (still ~30 speakers at
  0.6). When the speaker count is known, for instance 2-party telephony, setting
  `sherpa_num_clusters=2` brings it back down to the right count. Defaults are
  unchanged, so existing meeting-audio behaviour is preserved.

## [0.6.1] - 2026-07-21

### Fixed

- **Analyst time cadence now fires from a cold start.** `_on_utterance`
  measured the pending span with `newest_t1 - (self._oldest_pending_t0 or
  newest_t1)`, so a legitimate oldest-`t0` of `0.0` — every session's very first
  utterance — was treated as unset, collapsing the span to `0` and preventing
  `flush_every_s` from ever tripping until the queue rolled past that first
  utterance. Aligned the read with an explicit `is None` check (matching the
  sibling eviction guard). Regression test starts timestamps at `t0=0.0` and
  asserts the cadence still fires.

### Internal

- Error-path and edge-case test coverage for the analyst stage (prompt
  assembly + block fold, empty-queue no-op, Large Language Model (LLM)-failure keeps summary / drops
  poisoned block, Voice Activity Detection (VAD)-blip skip, time-vs-count cadence selection, run() drain +
  shutdown flush, missing-`[llm]` extra ImportError) plus VAD, sources, Language Identification (LID),
  API, and CLI. Coverage floor raised 35 → 55.
- Local CI-mirror gate: `make preflight` (pre-commit + ruff + tests) wired as a
  `pre-push` hook via `make install-hooks`.

## [0.6.0] - 2026-07-20

### Added

- **Transcript-viewer GUI at `GET /gui`.** A self-contained single page
  (HTML + Tailwind CDN + vanilla JS, no build step) served same-origin by the
  FastAPI app from the new in-package `vocal_helper.gui` module (mirrors the
  AI Helpers suite convention, e.g. `audio_helper.gui`). Drop an audio file
  **or paste a URL** → run the offline diarized pipeline → read a
  **speaker-labelled, colour-coded transcript** (one stable colour per speaker,
  with a live legend) alongside the rolling Gemma summary. Utterances reveal
  progressively (motion-guarded) so a long transcript reads as if it streams in.
  It POSTs to the existing `/pipeline` endpoint — zero new server logic — and
  contacts only the local server, so audio never leaves the machine. `GET /`
  now redirects to `/gui`. Because the module ships inside the package, the GUI
  works for a bare `pip install`, not only a source checkout.
- **URL ingest on `POST /pipeline`.** The endpoint now accepts an optional `url`
  form field (in addition to a file upload): the LOCAL server fetches any
  yt-dlp-reachable URL / RSS (Really Simple Syndication) / direct audio via `sources.from_url` (needs the
  `[stream]` extra). This powers the GUI's "paste a URL" affordance end-to-end.
  The uploaded `file` becomes optional; exactly one of `file` / `url` is required
  (a clean 400 otherwise).
- **Agent skill family `skills/vocal-helper/`** — a Claude Skill *and* an
  OpenCode skill (`SKILL.md` with a trigger-exhaustive third-person description
  + `references/{cli-reference,surfaces,triggers}.md`) plus `skills/README.md`
  with symlink-install instructions for `~/.claude/skills/` and
  `~/.opencode/skills/`.
- **`TRIGGERS.md`** — the exhaustive, user-facing catalogue of what invokes
  vocal-helper (transcription / diarization / "who spoke when" / summary /
  subtitles / language ID) and its SKIP boundary, referenced from README +
  LISEZMOI.
- **Local-first badge + `## The Promise` / `## La promesse`** sections in
  README / LISEZMOI, emphasising that voice and transcripts stay on-device.

### Changed

- The older static form GUI at `/ui` (repo `webui/` staticfiles) is kept for
  backward compatibility; `/gui` is now the canonical, in-package GUI.
- **`os-helper` pin lowered `>=1.5.3` → `>=1.5.2`** (the published floor). The
  diarization-bundle fetch prefers `os_helper.download_file` when present and
  falls back to a stdlib streaming download on older os-helper, so the pin stays
  satisfiable against PyPI without depending on an unpublished sibling release.

### Notes

- Public API unchanged — every name in `vocal_helper.__all__` is preserved. The
  new GUI, `/gui` route, and `url` form field are purely additive (SemVer minor).

## [0.5.2] - 2026-07-19

### Changed

- **Maturity classifier `3 - Alpha` → `4 - Beta`.** Reflects the actual state:
  104 functional tests at 50 % coverage, CI on Python 3.10–3.13, a corrected +
  regression-guarded router, and downstream use.
- **Pinned `pywhispercpp>=1.5`** — that line ships prebuilt wheels for macOS
  arm64, Linux x86_64/aarch64 (manylinux + musllinux) and Windows across
  cp39–cp314, so a plain `pip install vocal-helper` needs **no C/C++ toolchain**
  on any common platform (only exotic arches fall back to an sdist compile).
- **The hosted model bundle now carries the `sherpa` ONNX** (pyannote-3.0
  segmentation + TitaNet-large/small embedding), so the torch-free backend
  (`pip install vocal-helper[sherpa]`) runs offline diarization with **no
  PyTorch and no HuggingFace** — the lightest offline path. The bundle also
  dropped an unused Text-to-Speech (TTS) payload, so it stays lean.

### Documentation

- Added a **Versioning & stability** policy (README + LISEZMOI): pre-1.0, defaults
  change only in minors, never patches. Recorded honestly that the 0.5.1
  `--diar-backend` default change (`nemo` → `auto`) was part of *fixing the
  non-functional router*, not a gratuitous change — but such changes are
  minor-only from here.
- Clarified the install docs: the base install is light and wheel-based; the
  heavy `[nemo]` extra (~5 GB torch) and the model bundle are opt-in.

## [0.5.1] - 2026-07-19

### Fixed

- **The diarization router (the *aiguilleur*) is now actually enforced — it was
  decorative in 0.5.0.** The router shipped and printed a rationale but was never
  consulted with real inputs: the CLI called `select_diarization(duration_s=None)`
  (a constant → always pyannote on the auto branch), `--offline` bypassed it and
  used the `--diar-backend` default, the mic/url online path never called it, and
  `POST /pipeline` hardcoded `pyannote`. So the quality×speed **length crossover**
  that is the repo's headline value proposition could not fire in any production
  path. Now every surface routes on real conditions: files probe their duration
  (`vocal_helper.sources.probe_duration_s`, a cheap ffprobe read) so offline
  short (≤300 s, ≤4 spk) → `nemo` and long/unknown → `pyannote`; the API routes on
  the decoded buffer's real length; mic/url resolve the online backend (→ `nemo`).

### Changed

- **`--diar-backend` default `nemo` → `auto`** on both CLIs. `auto` delegates to
  the router (reporting Diarization Error Rate (DER) **and** Real-Time Factor (RTF)); an explicit `pyannote` / `nemo` /
  `sherpa` is honoured as an operator override. `POST /pipeline` `diar_backend`
  default `pyannote` → `auto` likewise.
- **`router.select_diarization` gained `nemo_available`** so a short file never
  routes to NeMo when the extra is absent — it falls through to pyannote rather
  than naming an unrunnable backend.

## [0.5.0] - 2026-07-19

### Added

- **Portable `sherpa` diarization backend (torch-free ONNX).** Selectable on both
  `OnlineDiarStage` (via `_SherpaEmbedder`) and `OfflineDiarStage` (via
  `_SherpaOfflineDiar`, sherpa-onnx `OfflineSpeakerDiarization`, whole-buffer). Runs the
  same TitaNet-large as the `nemo` backend through onnxruntime — no PyTorch, embeddable on
  every platform. Models resolve from `$VH_SHERPA_SEGMENTATION` / `$VH_SHERPA_EMBEDDING` or
  the diarization-engines bundle's `sherpa/` dir (community-1 segmentation preferred, our
  sovereign HF-free ONNX export). New `pip install vocal-helper[sherpa]` extra and
  `tests/test_diar_sherpa.py`. Study-selected in the `pasdebonneoudemauvaisesituation`
  sandbox (ADR 0002): DER 0.174 on AMI ES2011a, 0.148 on held-out IS1008a, FR+EN validated;
  streaming = periodic offline re-diarization.
- **`sherpa` selectable from the CLI.** Both front-ends now offer
  `--diar-backend sherpa` (argparse and click) — previously the backend
  existed and was tested but was unreachable from the command line
  (`choices` listed only `pyannote`/`nemo`).
- **`transcribe_pcm_with_language` helper** (`vocal_helper.asr`) returns
  `(text, detected_language)` so callers can report the language whisper
  actually used instead of echoing the request.
- **Backend router — the *aiguilleur* (`vocal_helper.router`).**
  `select_diarization(live, duration_s, max_speakers, torch_free,
  pyannote_available)` returns a `BackendPlan(mode, backend, expected_der,
  expected_rtf, reason)` that turns the measured quality×speed trade-off into one
  explicit, tested decision — both CLIs delegate to it via `_choose_file_diar`,
  so no front-end hard-codes a backend. Quality (DER) and speed (RTF) are
  first-class fields keyed from a single `_PROFILE` table, re-validated
  on-machine (`studies/router_profile_validation.py`, `pyannote.metrics` collar
  0.25) against bagarre + AMI. Findings: offline has a length crossover (nemo
  short ≤ 4 spk ↔ pyannote long/unknown/> 4 spk); online has none, so streaming
  always routes to nemo (torch-free → sherpa). `router.py` is 100 % covered
  (`tests/test_router.py`).

### Changed

- **Surfaces reshaped to a pure toolbox.** vocal-helper now exposes only
  library + argparse CLI + click CLI + FastAPI HTTP API + Model Context Protocol (MCP). The API and MCP
  surfaces are **kept**; the `POST /transcribe` response now carries the detected
  language, and `api.py` reads `__version__` dynamically (no more drift).

- **Language identification is now discovery-first — no default, no pairing.**
  `vocal_helper.lid` detection (`detect_language`, `language_posterior_curve`,
  `detect_language_regions{,_fast}`, `detect_language_speechbrain`,
  `cross_check_regions`) defaults to `supported=None`, meaning *return the
  language the input actually is* (whisper's true argmax / VoxLingua107's true
  label). A restricted candidate set is now an **opt-in routing hint**, never a
  default: `DEFAULT_SUPPORTED_LANGS` is demoted to a convenience constant with
  no privileged language. Empty or too-short-to-identify audio returns **no
  region** rather than inventing English. Completes the direction set by the
  earlier `lang_pair` removal ("language is discovered, not paired").

### Fixed

- **`POST /transcribe` reported the detected language, not the request echo.**
  With `language=auto` the endpoint now returns the language whisper detected
  from the audio instead of literally echoing `"auto"` back.
- **FastAPI app version was pinned at `0.3.7`** (four releases stale) — now
  tracks the package version (`0.4.7`).
- **`__email__` typo** (`warithmetics@deraison.ai` → `warithmetics@deraison.ai`) and
  stale `__version__` / `CITATION.cff` (0.4.6 → 0.4.7) brought in line with
  `pyproject.toml`.

### Removed

- **Docker (`Dockerfile`) and the `GUI.md` product plan.** vocal-helper is a
  toolbox (library + CLIs + API + MCP), not a shipped container or visual
  product. Containerization stays available as self-authored deployment recipes
  in `TECHNICAL_STACK.md`; no Dockerfile is shipped.

## [0.4.7] - 2026-07-18

### Documentation

- Complete type annotations and Numpy-style docstrings for CLI/api/pipeline
  closures and the language-id helper per CODING.md.

## [0.4.6] - 2026-07-16

### Fixed

- **Offline NeMo Sortformer backend returned no speakers at all.**
  `OfflineDiarStage(backend="nemo")` parsed the model output as legacy Rich Transcription Time Marked (RTTM)
  (`SPEAKER …` lines, ≥8 fields), but nemo-toolkit 2.x emits the compact
  `"<start> <end> <speaker>"` form — so every line was dropped and the backend
  produced an empty diarization (DER 1.0 on every input). The parser now
  handles both formats (extracted to `_parse_sortformer_segments`, unit-tested
  in `tests/test_sortformer_parse.py`). The offline nemo path now works.

### Documentation

- **Offline backend crossover, measured (DER, collar 0.25).** With Sortformer
  fixed, a head-to-head on ground truth shows a length-dependent crossover:

  | corpus | offline pyannote | offline nemo (Sortformer) |
  |---|---|---|
  | AMI (20–40 min meetings) | **0.122** | 0.242 |
  | bagarre (~30 s, ≤4 spk, 26% overlap) | 0.338 | **0.177** |

  NeMo Sortformer (`diar_sortformer_4spk-v1`, end-to-end + overlap-aware, but
  4-speaker / ~90 s capped) nearly *halves* the DER on short ≤4-speaker clips,
  while pyannote 3.1 (whole-buffer, no speaker cap) wins on long meetings. So
  **pyannote stays the offline default** (robust across length + speaker count,
  best on the long inputs that dominate batch `file` use) and `--offline
  --diar-backend nemo` is the pick for short ≤4-speaker workloads. Documented
  in the `vocal_helper.diar` module docstring; `studies/diar_der_paths.py`
  reproduces the full sweep.

## [0.4.5] - 2026-07-16

### Changed

- **Batch file diarization now defaults to the reliable offline path.** A
  real-DER sweep against ground truth (`studies/diar_der_paths.py`,
  pyannote.metrics, collar 0.25) settled which diarizer to trust:

  | corpus | offline pyannote | online (no refine) | online + refine |
  |---|---|---|---|
  | AMI (real meetings) | **0.122** | 0.497 | 0.351 |
  | bagarre (26% overlap) | **0.338** | 0.586 | 0.592 |

  Offline pyannote is literature-grade (Bredin 2023 ≈ 0.188 uncollared) and
  ~3× lower DER than the online streaming diarizer, which cannot model
  overlapped speech. So `vocal-helper file --no-real-time` now **auto-selects
  the offline pyannote diarizer when its bundle is present**, falling back to
  the online diarizer + `refine_on_close` repair pass when it is not (with a
  one-line stderr note either way). New `--online` flag forces the streaming
  diarizer; explicit `--offline` is unchanged and still honours
  `--diar-backend`. Live `mic` / `url` are untouched. Downstream integrators
  embedding diarization in a larger pipeline should use `OfflineDiarStage` /
  `OfflinePipeline` for batch audio — see the `vocal_helper.diar` docstring.
  The DER sweep also independently confirms the v0.4.4 `refine_on_close` fix:
  it roughly halves the online DER on meetings that over-segment (ES2011a
  0.588 → 0.296) and never hurts. Both CLIs (argparse + click) share one
  `_choose_file_diar` policy so the default cannot drift; covered by
  `tests/test_cli_diar_default.py`.

## [0.4.4] - 2026-07-16

### Fixed

- **Online diarizer over-segmentation on long batch files.** The greedy
  single-pass online clusterer (`OnlineDiarStage`) mints a permanent new
  speaker whenever an embedding is farther than `join_threshold` from every
  existing centroid, with no cap and no merge — so on long multi-speaker
  audio, outlier embeddings (overlap, laughter, jingle, backchannels, slow
  centroid drift) each spawn a throwaway singleton, producing hundreds of
  spurious speaker labels for a handful of real speakers. `OnlineDiarStage`
  now supports a `refine_on_close` batch pass that, once the stream ends,
  globally re-clusters the collected per-segment embeddings — merging
  near-duplicate centroids (`merge_threshold`) and pruning micro-clusters
  smaller than `min_cluster_size` into their nearest surviving speaker —
  then emits the batch with corrected, compact `S<n>` labels. `vocal-helper
  file --no-real-time` (both the argparse and click CLIs) enables this
  automatically; live streaming is unchanged. An optional online
  `max_speakers` cap is also available. Covered by new model-free
  regression tests (`tests/test_diar_refine.py`).

## [0.4.3] - 2026-07-15

### Documentation

- Harmonize README/LISEZMOI to the AI Helpers common structure (single H1,
  Documentation block, PyPI + source install paths, refreshed pins to v0.4.3);
  no code changes.

## [0.4.2] - 2026-07-14

### Added

- `lid.detect_language_regions_fast` — a single-pass fast path for spoken-language
  segmentation. It runs one cheap whole-file `detect_language`; when that clears a
  confidence gate (`DEFAULT_FAST_CONF_GATE`, 0.5) on a routable language the file is
  treated as monolingual and returned as a single region, otherwise it falls back to
  the accurate posterior-curve `detect_language_regions` scan. On monolingual
  recordings this cuts per-file language identification from ~73 s to ~1 s with
  identical region output, while low-confidence (code-switched / noisy) files still
  get the full scan. Exported from the package root; covered by pure, model-free tests.

### Documentation

- Finalize suite wording: describe capabilities in plain language
  (Voice Activity Detection, Speech to Text, Speech Synthesis, source
  separation) instead of specific tool names, for consistency across the
  suite's descriptions and the documentation site.


## [0.4.1] - 2026-07-14

### Maintenance

- Apply the project coding standards across the package and `tests/`:
  Numpy-style docstrings on every function/class (including private and
  nested helpers), full type annotations with `from __future__ import
  annotations`, and comment density raised above the floor in every
  module. No public API or behavior changes.
- Route library logging through the os-helper logging surface
  (`osh.info/warning/error`) and adopt os-helper path/file utilities
  more widely; pin `os-helper>=1.5.0`.
- Refresh the project logo asset.


## [0.4.0] - 2026-07-14

### Changed

- **Offline diarization now runs whole-buffer by default.** The pdbms offline
  map-reduce study (2026-07-14 — full stack VAD + Automatic Speech Recognition (ASR) + diar on AMI, scored by
  VAD F1 / Word Error Rate (WER) / DER-JER against AMI ground truth) found DER strictly monotone
  in chunk size: whole-buffer is best (median DER **0.143** vs 0.170 at 300 s,
  and cliffs to 0.31 / 0.50 at 120 s / 60 s as speaker fragmentation outruns
  the stitch); VAD is chunk-invariant; and ASR *destabilises* when chunked (one
  meeting hit WER 1.17 on a long-window whisper loop). Accordingly
  `IDEAL_DURATION_S_PYANNOTE` is raised **300 → 3600 s**: any realistic input
  (≤ 1 h) now diarizes as a single whole-buffer call, and chunk-and-stitch
  survives only as a memory backstop past ~1 h. **NeMo is unchanged at 60 s**
  (its Sortformer 90 s cap forces chunking). Override per call with
  `OfflinePipelineConfig(diar={"ideal_duration_s": …})`.

- **No HuggingFace needed at runtime.** All model weights that used to be
  pulled from the HF hub now load from a self-hosted, HF-free bundle
  (`diarization-engines.zip`) — resolved via `resolve_diarization_engines()`
  from `$VH_DIARIZATION_ENGINES` (a local dir or URL) or the built-in default,
  cached locally and verified against a manifest. Covers the offline pyannote
  3.1 pipeline (local `config.yaml` + `.bin` weights), NeMo Sortformer (local
  `.nemo` via `restore_from` — also fixes the wrong `diar_sortformer_v1` id →
  `diar_sortformer_4spk-v1`), the online `pyannote/embedding` embedder, and the
  SpeechBrain VoxLingua107 language-ID cross-check. TitaNet already loads from
  NGC (no HF). Set `$HF_HUB_OFFLINE=1` and the full stack runs with no token,
  no HF network access. HF remains only as an automatic fallback when no bundle
  is configured.

### Added

- **Offline quality regression test (DeepEval).** `tests/test_offline_regression.py`
  runs the offline pyannote + whisper.cpp stack on a small hosted AMI subset
  (CC BY 4.0) and asserts DER / WER thresholds via custom DeepEval
  `BaseMetric`s. Marked `integration` (skipped unless models are present).

### Removed

- **`vocal_helper.tts` (PiperTTS) dropped.** Text-to-speech is out of scope for
  a diarization + transcription library and is covered far better by the
  dedicated `speaker-helper` / `speak` projects. Removes the `PiperTTS` export,
  the `[tts]` extra (`piper-tts`), and the `rhasspy/piper-voices` HF download.
  **Breaking:** import `PiperTTS` from those projects instead.

## [0.3.7] - 2026-07-13

### Changed

- **`vocal_helper.lid` is now a posterior-curve language diarizer.**
  `detect_language_regions` no longer classifies fixed windows; it samples a
  language-posterior curve over overlapping windows
  (`language_posterior_curve`), Gaussian-smooths it, takes the per-frame
  argmax, absorbs short regions, locally refines each change point (fine
  re-scan + interpolated crossing) and snaps it to the nearest silence — a
  language switch has no acoustic boundary, so it is classified and smoothed,
  not detected (after Wang et al., Interspeech 2019). Boundaries resolve to
  ~±0.4 s on a clean code-switch vs the previous ±window/2. Defaults changed:
  `DEFAULT_WINDOW_S` 20→10, `DEFAULT_MIN_REGION_S` 10→8.

### Added

- **Independent verification.** `cross_check_regions` /
  `detect_language_speechbrain` corroborate each region with a second, fully
  independent audio classifier — SpeechBrain VoxLingua107 ECAPA-TDNN — that
  shares nothing with whisper but the signal. Optional `[lid]` extra
  (`speechbrain`, `torchaudio`), imported lazily. New public exports:
  `language_posterior_curve`, `detect_language_speechbrain`,
  `cross_check_regions`, `RegionVerdict`.

## [0.3.6] - 2026-07-12

### Added

- **Acoustic language identification (`vocal_helper.lid`).** New module for
  language diarization off the same cached whisper.cpp model
  `WhisperStage` uses: `detect_language(pcm)` returns the most probable
  ISO-639-1 language among a configurable `supported` set (restricting the
  candidates keeps whisper from ranking a close relative — Galician/Catalan
  over Spanish — on short windows), and `detect_language_regions(pcm, sr)`
  slides that detection over the audio, coalesces same-language neighbours
  and absorbs regions shorter than `min_region_s` into their longer
  neighbour, returning `LangRegion(lang, t0, t1)`. Enables per-region
  transcription of code-switching recordings, where a naive whisper `"auto"`
  pass would lock onto one language and mistranscribe the rest. Exported
  from the package root.

### Changed

- Re-pinned sibling helpers to their newest tags: os-helper `v1.4.2`,
  audio-helper `v1.5.5`, capture-helper `v0.2.2`, podcast-helper `v0.3.3`.

## [0.3.5] - 2026-07-09

### Added

- **Offline full-throttle ASR batching.** `WhisperStage` gained
  `batch` + `max_chunk_s` keyword args. When `batch=True` it packs
  consecutive diarized segments into ≤ `max_chunk_s` (default 24 s)
  windows and runs **one whisper call per window** instead of one per
  segment, re-mapping each phrase back to its segment by local time
  window. whisper.cpp pads every call to a 30 s mel, so fewer/fuller
  calls amortise the fixed encoder cost. The 2026-07-09 sweep
  (`studies/asr_offline_batching.py`, 12 bilingual bagarre-rich mixes)
  measured **6.5× lower RTF (0.054 vs 0.353) at *better* WER (0.565 vs
  0.612)** — the longer decoder context cuts short-segment hallucination.
  `OfflinePipeline` enables it by default; opt back into the per-segment
  path with `OfflinePipelineConfig(asr={"batch": False})`.
- **Streaming warm-up.** `WhisperStage(warmup=True)` runs one throwaway
  inference on silence before consuming the queue, moving whisper's
  cold-start stall off the first caption. `Pipeline` (streaming) enables
  it by default.

### Notes

- Both features are additive and default-off at the `WhisperStage`
  level; the public API surface (exports, `Utterance`/`DiarizedSegment`
  keys, `S0`/`S1`/`S?` speaker labels, stage signatures) is unchanged and
  now covered by `tests/test_public_api_stability.py`. Offline still emits
  exactly one `Utterance` per diarized segment.

## [0.3.4] - 2026-07-08

### Documentation

- Cross-platform Install prerequisites (macOS / Ubuntu / Windows).

## [0.3.3] - 2026-07-07

### Fixed

- CI now installs ffmpeg (required by the ffmpeg-based load_audio) so the
  from_wav_file tests run. No package change vs 0.3.2.

## [0.3.2] - 2026-07-07

### Changed

- Decode audio via `audio_helper.load_audio` (ffmpeg) everywhere — API,
  both CLIs and `from_wav_file` now accept ANY format (mp3, m4a/AAC,
  opus, video audio tracks) and auto-resample; `from_wav_file` no longer
  requires a 16 kHz WAV. Diar/TTS writes go through `scipy.io.wavfile`.
- Remove `soundfile` entirely (code + extras); add `scipy` as a direct dep.
- Bump pins: audio-helper v1.5.4, podcast-helper v0.3.2. Bump 0.3.1 -> 0.3.2.

## [0.3.1] - 2026-07-07

## [0.3.0] — 2026-07-06

### Added

- `OnlineDiarStage` and `OfflineDiarStage` now accept a `device`
  kwarg (`"cpu"` / `"cuda"` / `"mps"` / `None`). Default `None`
  uses the new `_auto_torch_device` helper which picks CUDA > MPS >
  Central Processing Unit (CPU). On Apple Silicon this lifts pyannote 3.1 from ~ 15× real-time
  (CPU) to roughly real-time (MPS). `_PyannoteOfflineDiar.load`
  wraps `pipeline.to(...)` in a `try` block ; if MPS rejects an op
  the stage stays on CPU rather than crash. `_PyannoteEmbedder`
  forwards the device to `pyannote.audio.Inference(..., device=)`
  the same way. The previously-skipped
  `test_offline_pipeline_vs_youtube_captions` integration test is
  re-enabled.
- `vocal_helper._settings` — hand-rolled `settings.yaml` loader (no
  PyYAML dep) and `resolve_hf_token()` helper. The CLI, the example
  script, and `OnlineDiarStage` / `OfflineDiarStage` now resolve the
  HuggingFace token in this order: explicit value > `HF_TOKEN` env
  var > `secrets.hf_token` in `settings.yaml`.
- `settings.yaml.example` — git-tracked template; copy to
  `settings.yaml` (git-ignored) and fill in the real token. The
  placeholder `hf_XXXX` is treated as unset.
- Override the lookup with `$VOCAL_HELPER_SETTINGS` to pin a specific
  file (handy for tests and unusual deploys).
- Test suite expanded from 3 → 40 cases — split into
  `tests/test_smoke.py`, `test_settings.py`, `test_sources.py`,
  `test_pipeline.py`, `test_cli.py`. A `conftest.py` autouse fixture
  isolates `$HF_TOKEN` / `$VOCAL_HELPER_SETTINGS` so the developer's
  real `settings.yaml` cannot leak into CI assertions.
- `.github/workflows/ci.yml` hardened with `lint`
  (ruff check + ruff format --check informational), `test` (Python
  matrix 3.10/3.11/3.12/3.13 + coverage report on 3.12), and
  `pre-commit` (mirror of the local commit-time hooks). Adds pip
  caching and per-ref concurrency cancellation.
- Whisper **bias prompt** — `WhisperStage(initial_prompt=…)` and
  `transcribe_pcm(initial_prompt=…)`, surfaced as `--initial-prompt`
  on every CLI. Empty by default; a domain-aligned prompt cut WER
  15–25 pp and saved up to 39 % RTF on the 2026-06-30 AMI sweep.
- **`SemanticEOTStage`** (`vocal_helper.eot`) — opt-in, LiveKit-style
  turn detector that holds back VAD segments that look mid-thought and
  merges them with their successor. Enable with `PipelineConfig(eot=…)`
  or `--eot` / `--eot-model`; off by default (one extra LLM hop per
  voiced segment). Offline path deliberately has no EOT block.
- **Multi-surface exposure** — the same pipeline is now reachable via
  the argparse CLI (`vocal-helper`), a click twin (`vocal-helper-click`),
  a FastAPI HTTP app (`vocal_helper.api`), and an MCP server
  (`vocal-helper-mcp`), all sharing one config builder.
- `vocal_helper.tts` (local Piper TTS) and a
  `vocal_helper.parallel_pipelines` demonstrator.

### Changed

- Pre-existing ruff lint errors in `vocal_helper/diar.py`,
  `vocal_helper/pipeline.py`, `vocal_helper/sources.py` and
  `vocal_helper/vad.py` cleaned up (unused `step` / `frame_period_s`
  vars, missing `strict=` on `zip`, six try/except/pass blocks
  rewritten as `contextlib.suppress`, import-order). `ruff check .`
  now passes from a clean checkout.
- Ruff exclusion list moved from `tests/` to `studies/` — one-off
  research scripts are no longer linted, but tests now are.
- **Default online-diar backend** switched `pyannote` → `nemo`
  (TitaNet) per the 2026-06-30 embedding sweep (+76 % separability
  margin on AMI). Pass `backend="pyannote"` to skip the ~5 GB NeMo
  install.
- **Default LLM analyst model** switched `gemma4:e4b` → `gemma3:4b`
  per the 2026-06-30 7-model Pareto sweep (3× faster *and* higher
  cos_sim). Reflected across the library and every CLI surface.
- `vocal_helper.cli` is now a thin backward-compat shim; the canonical
  argparse implementation lives in `vocal_helper.cli_argparse`, so the
  four surfaces share a single config builder with no drift.
- Removed the attribution-audit CI job and pre-commit hook;
  `nomoreclaude.sh` is no longer tracked (kept as a local-only tool).

## [0.1.0] — 2026-06-30

Initial release.

### Added

- `vocal_helper.types` — `PcmFrame`, `VoicedSegment`, `DiarizedSegment`, `Utterance`, `SummarySnapshot` typed dicts.
- `vocal_helper.sources` — `from_microphone` (capture-helper), `from_wav_file`, `from_numpy_array`.
- `vocal_helper.vad.SileroVADStage` — Silero v5 ONNX VAD with run-by-run emission and edge-padded segments.
- `vocal_helper.diar.OnlineDiarStage` — online cosine-distance running-mean clustering, `pyannote` and `nemo` backends, `join_threshold=0.30` default.
- `vocal_helper.asr.WhisperStage` — pywhispercpp turbo wrapper with thread-pool execution and word timestamps.
- `vocal_helper.llm.GemmaAnalystStage` — Ollama-served `gemma4:e4b` rolling-summary analyst, recent window 60 s, summarises every 5 evicted utterances.
- `vocal_helper.pipeline.Pipeline` — top-level orchestrator with `subscribe_voiced` / `subscribe_diarized` / `subscribe_utterances` fan-out hooks.
- `vocal-helper` CLI with `mic` and `file` subcommands.
- `examples/live_mic_to_text.py` — end-to-end live demo.
- Smoke tests in `tests/test_smoke.py`.
