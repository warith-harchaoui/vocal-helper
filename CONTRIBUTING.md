# Contributing to Vocal Helper

Thanks for picking this up.

Before you file an issue or open a PR, please read
[`SECURITY.md`](SECURITY.md) if the report is security-sensitive, and
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for community expectations.
Deployment / production questions belong in
[`TECHNICAL_STACK.md`](TECHNICAL_STACK.md), and every non-obvious
default is motivated by a script under [`studies/`](studies/README.md).

## Dev install

```bash
git clone https://github.com/warith-harchaoui/vocal-helper.git
cd vocal-helper
pip install -e '.[dev,all]'
```

## Pre-commit hooks

The repo ships a `.pre-commit-config.yaml` that mirrors the CI gates
locally. Install once :

```bash
pip install pre-commit
pre-commit install                     # runs on every ``git commit``
pre-commit install --hook-type pre-push  # runs pytest + attribution audit before push
```

The hooks are :

- **Commit time (fast)** — `ruff --fix`, trailing whitespace,
  end-of-file newline, YAML / TOML syntax, merged large files, LF
  line endings.
- **Push time (slower)** — `pytest -q` (unit only, no integration)
  and `bash nomoreclaude.sh` (attribution audit).

Run manually against the whole tree :

```bash
pre-commit run --all-files
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
