<!--
Thanks for the PR! A few checks the reviewer will run before merging.
Fill in what applies — delete the rest.
-->

## Summary

<!-- One or two sentences. What's changing and why. -->

## Type of change

- [ ] `feat` — new user-visible capability
- [ ] `fix` — bug fix
- [ ] `refactor` — no behaviour change
- [ ] `perf` — measured RTF / memory improvement
- [ ] `docs` — README / LISEZMOI / EXAMPLES / TECHNICAL_STACK / CHANGELOG
- [ ] `test` — new tests, or CI hardening
- [ ] `chore` — dep bump, tooling, non-code

## Impact

- [ ] Public API changed (add / remove / signature edit) — noted in `CHANGELOG.md` Unreleased.
- [ ] Default backend / model / device changed — study log linked below.
- [ ] Extra pulled or dropped in `pyproject.toml`.
- [ ] Behaviour observable via logs / metrics changed.

## Study log

<!--
If this PR changes a default (backend, model, threshold, cadence…),
link the study under studies/ that motivated the choice. New defaults
without a study need reviewer sign-off.
-->

- Study path : `studies/`_______________
- Corpus : ______________
- Metric that moved : __________ (from ___ to ___)

## Test evidence

```
$ pytest -q
<paste the last 5 lines>
```

```
$ ruff check .
<paste "All checks passed!" or the fixes applied>
```

## Deployment note

<!--
If ops need to do anything when this ships (new secret, model pull,
env var, cache invalidation), spell it out in one line.
-->

## Reviewer checklist

- [ ] Public API changes are documented in the docstring **and** listed in CHANGELOG.
- [ ] New heavy runtime deps are gated behind an extra.
- [ ] Tests fail without the change and pass with it.
- [ ] No AI-attribution mentions anywhere (author / committer / message / trailers / files).
- [ ] `TECHNICAL_STACK.md` reference table still holds if the RTF envelope moved.
