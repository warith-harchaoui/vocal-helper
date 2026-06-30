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

The integration suite expects :

- `HF_TOKEN` exported for the pyannote fetch ;
- `ollama serve` running locally with `gemma4:e4b` pulled ;
- a microphone reachable through `capture_helper.list_sources("microphone")`.

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
