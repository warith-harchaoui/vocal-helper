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

## Local gate: catch red CI before you push

Everything the server enforces is mirrored by a single local target, so
failures surface on your machine instead of blocking on GitHub:

```bash
make preflight     # pre-commit hooks + ruff check + ruff format-check + pytest
```

A green `preflight` means the CI pipeline in `.github/workflows/ci.yml`
will pass. Wire it as an enforced pre-push gate (you can't push red):

```bash
make install-hooks   # writes a pre-push hook that runs `make preflight`
```

`make install-hooks` is deliberately robust to a **global**
`core.hooksPath` (some setups route all hooks to `~/.config/git/hooks`,
e.g. for an anti-attribution `prepare-commit-msg`). Because git honours
only one hooks directory, the script points `core.hooksPath` at
`.git/hooks` for this repo only (leaving global config untouched) and
copies any pre-existing global hooks in so they keep firing here. Bypass
the gate in a pinch with `git push --no-verify`.

### The underlying hooks

The repo ships a `.pre-commit-config.yaml` that mirrors the CI gates:

- **Commit time (fast)**: `ruff --fix`, trailing whitespace,
  end-of-file newline, YAML / TOML syntax, merged large files, LF
  line endings.
- **Push time (slower)**: `pytest -q` (unit only, no integration).

Run the hygiene + ruff hooks against the whole tree:

```bash
make precommit                # → pre-commit run --all-files
```

> If your machine does **not** set a global `core.hooksPath`, the stock
> `pre-commit install && pre-commit install --hook-type pre-push` also
> works; `make install-hooks` is the portable path that survives either
> setup.

## Tests

```bash
pytest -q
```

Default run skips integration tests (the ones that load Whisper, pyannote, NeMo, or talk to Ollama). To exercise them:

```bash
pytest -q -m integration
```

The unit suite is fast (< 100 ms) and split by surface:

- `tests/test_smoke.py`: top-level imports, frame shapes, config sanity.
- `tests/test_settings.py`: the `settings.yaml` parser and its
  explicit-path / env-var / file resolution precedence.
- `tests/test_sources.py`: `from_numpy_array` / `from_wav_file` contracts.
- `tests/test_pipeline.py`: config defaults, queue sizing, stage validation.
- `tests/test_cli.py`: argparse + click CLI surfaces, `_build_config`
  end-to-end wiring (including the LLM engine block), router backend
  resolution.

`tests/conftest.py` strips `HF_TOKEN` and `VOCAL_HELPER_SETTINGS` from
every test so a developer's local `settings.yaml` cannot leak into
assertions. Opt-in by re-setting the env var inside the test.

Coverage report on demand:

```bash
pytest --cov=vocal_helper --cov-report=term-missing
```

The integration suite expects:

- a local `settings.yaml` (copy `settings.yaml.example` to bootstrap) so
  the diarization-engines bundle resolves and downloads (no HuggingFace
  token needed: the bundle is self-hosted and unlocks with no auth);
- an LLM backend up and reachable at whatever `voh.resolve_engine()`
  picks for the machine (Ollama on macOS/CPU, vLLM on a discrete GPU);
- a microphone reachable through `capture_helper.list_sources("microphone")`.

## CI

Three jobs run on every push and PR to `main`
(see `.github/workflows/ci.yml`):

- **lint**: `ruff check .` (hard fail) + `ruff format --check .`
  (informational until adopted).
- **test**: pytest across Python 3.10 → 3.13 with pip caching;
  coverage XML uploaded as an artifact on the 3.12 leg.
- **pre-commit**: `pre-commit run --all-files`, mirroring the local
  commit-time hooks (ruff + filesystem hygiene).

## Lint / format

```bash
ruff check .
ruff format .
```

`ruff` rules and ignores live in `pyproject.toml`.

## Code style

- Async first: every public coroutine entry is `async def`. Blocking work (whisper.cpp, Ollama generate) wraps with `asyncio.to_thread`.
- One responsibility per stage. If a stage grows another reason to change, split it.
- Defensive on backend failures, opinionated on shapes: a malformed PCM array should fail loudly at construction, not three stages later.
- Comments explain *why*, not *what*. Names carry the *what*.

## Commit messages

We follow Conventional Commits:

- `feat(diar): online cosine clusterer with EMA centroid updates`
- `fix(vad): emit edge_pad_ms even when the run ends on the lead pad`
- `docs: rewrite README diarization section after stitch-threshold sweep`
