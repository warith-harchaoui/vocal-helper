---
name: Feature request
about: Propose a new stage, backend, extra, or public API.
title: "feat: "
labels: ["enhancement", "needs-triage"]
assignees: []
---

## What problem are you solving?

<!--
A concise description of the pain. Prefer real user stories over
abstract "would be nice to have" — the design will be better if we
know the concrete failure mode being avoided.
-->

## Proposed shape

<!--
Sketch the public API you would like to call. Signature + a 5-line
example. If a similar pattern exists elsewhere in the suite
(vocal_helper.sources.from_url, OnlineDiarStage, …), reference it.
-->

```python
# your proposed API
```

## Alternatives considered

<!--
What else would work? Why is the proposed shape better?
Mention adjacent-project prior art (LiveKit Agents, Pipecat, NeMo,
pyannote, whisper.cpp, …) if it informed the design.
-->

## Scope / trade-offs

- Does this change the CPU / GPU footprint at runtime? By how much?
- Does it add a new heavy dependency? If so, which extra should gate it?
- Does it need a matching backend study under ``studies/`` before merge?

## Checklist

- [ ] I read the existing `sources.py` / `pipeline.py` and this doesn't overlap.
- [ ] The proposed API is discoverable via the public `__all__`.
- [ ] The proposal doesn't ask the library to hide device / auth complexity
      from callers who legitimately need to control it.
