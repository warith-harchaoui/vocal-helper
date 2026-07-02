# Contributing to Vocal Helper

Thanks for picking this up.

## Dev install

```bash
git clone https://github.com/warith-harchaoui/vocal-helper.git
cd vocal-helper
pip install -e '.[dev,all]'
```

## Tests

```bash
pytest -q
```

Default run skips integration tests (the ones that load Whisper, pyannote, NeMo, or talk to Ollama). To exercise them :

```bash
pytest -q -m integration
```

The unit suite is fast (< 100 ms) and split by surface :

- `tests/test_smoke.py` — top-level imports, frame shapes, config sanity.
- `tests/test_settings.py` — YAML loader and `resolve_hf_token` precedence.
- `tests/test_sources.py` — `from_numpy_array` / `from_wav_file` contracts.
- `tests/test_pipeline.py` — config defaults, queue sizing, stage validation.
- `tests/test_cli.py` — argparse + `_build_config` HF-token resolution.

`tests/conftest.py` strips `HF_TOKEN` and `VOCAL_HELPER_SETTINGS` from
every test so a developer's local `settings.yaml` cannot leak into
assertions. Opt-in by re-setting the env var inside the test.

Coverage report on demand :

```bash
pytest --cov=vocal_helper --cov-report=term-missing
```

The integration suite expects :

- `HF_TOKEN` exported for the pyannote fetch (or `secrets.hf_token`
  in a local `settings.yaml` — copy `settings.yaml.example` to
  bootstrap) ;
- `ollama serve` running locally with `gemma4:e4b` pulled ;
- a microphone reachable through `capture_helper.list_sources("microphone")`.

## CI

Three jobs run on every push and PR to `main`
(see `.github/workflows/ci.yml`):

- **lint** — `ruff check .` (hard fail) + `ruff format --check .`
  (informational until adopted).
- **attribution-audit** — runs `nomoreclaude.sh` in audit mode. Hard
  fail if any commit subject/author or tracked file matches the
  unwanted AI-attribution regex encoded inside the script.
- **test** — pytest across Python 3.10 → 3.13 with pip caching;
  coverage XML uploaded as an artifact on the 3.12 leg.

## Lint / format

```bash
ruff check .
ruff format .
```

`ruff` rules and ignores live in `pyproject.toml`.

## Code style

- Async first — every public coroutine entry is `async def`. Blocking work (whisper.cpp, Ollama generate) wraps with `asyncio.to_thread`.
- One responsibility per stage. If a stage grows another reason to change, split it.
- Defensive on backend failures, opinionated on shapes : a malformed PCM array should fail loudly at construction, not three stages later.
- Comments explain *why*, not *what*. Names carry the *what*.

## Commit messages

We follow Conventional Commits :

- `feat(diar): online cosine clusterer with EMA centroid updates`
- `fix(vad): emit edge_pad_ms even when the run ends on the lead pad`
- `docs: rewrite README diarization section after stitch-threshold sweep`
