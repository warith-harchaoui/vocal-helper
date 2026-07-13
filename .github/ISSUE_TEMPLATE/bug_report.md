---
name: Bug report
about: A stage misbehaves, a test fails on your setup, an example doesn't run.
title: "bug: "
labels: ["bug", "needs-triage"]
assignees: []
---

## Summary

<!-- One sentence. What's broken? -->

## To reproduce

Minimal repro — the smallest snippet that shows the problem :

```python
# your snippet here
```

Or the exact command line + input file :

```bash
vocal-helper file bad.wav --offline --llm
```

## Expected behaviour

<!-- What did you expect to happen? -->

## Actual behaviour

<!-- What happened instead? Paste the traceback in full if there is one. -->

```
<paste stderr / traceback here>
```

## Environment

Please paste the output of the following one-liner :

```bash
python -c "
import sys, platform, importlib.metadata as m
for p in ('vocal-helper', 'torch', 'pyannote.audio', 'pywhispercpp',
          'silero-vad', 'ollama', 'audio-helper', 'podcast-helper'):
    try:
        print(f'{p}\t{m.version(p)}')
    except m.PackageNotFoundError:
        print(f'{p}\tNOT INSTALLED')
print(f'python\t{sys.version.splitlines()[0]}')
print(f'platform\t{platform.platform()}')
"
```

```
<paste output here>
```

GPU / device (relevant for pyannote + whisper) :

- [ ] CPU only
- [ ] Apple Silicon (MPS)
- [ ] NVIDIA CUDA — model : ______
- [ ] Other : ______

## Checklist

- [ ] I searched existing issues and this isn't a duplicate.
- [ ] I ran on the latest ``main`` commit (or specified the exact SHA).
- [ ] The traceback is included in full (no ellipsis).
