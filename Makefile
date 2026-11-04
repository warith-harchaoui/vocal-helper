# Local development gates — a one-command mirror of the server CI.
#
# The point: run `make preflight` (wired as a git pre-push hook by
# `make install-hooks`) and a green result means the GitHub Actions run in
# .github/workflows/ci.yml is green too. Failures surface here, on your machine,
# instead of blocking on the server — which is what lets the server CI stay a
# single light job that confirms rather than discovers.
#
# Every hard gate the server enforces has a target here with the SAME strictness,
# and `preflight` runs the union. They drifted once: the format check was
# informational here (`-ruff format --check`) while the server blocked on it, so a
# green preflight could still be a red push.

PYTHON ?= python
RUFF   ?= ruff
PYTEST ?= pytest
# The `requires-python` floor. The server installs and imports on it; locally the
# check runs when an interpreter of that version is on PATH and otherwise falls
# back to a syntax-level check at that target, said out loud.
FLOOR_PYTHON ?= python3.10

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: sync
sync: ## Install the package + dev extra exactly as the server does
	$(PYTHON) -m pip install -e '.[dev]'

.PHONY: env-check
env-check: ## The installed environment satisfies pyproject.toml (base + dev)
	$(PYTHON) scripts/check_env.py

.PHONY: lint
lint: ## ruff check — hard gate, as on the server
	$(RUFF) check .

.PHONY: format-check
format-check: ## ruff format --check — hard gate, as on the server
	$(RUFF) format --check .

.PHONY: test
test: ## pytest + coverage floor, integration deselected — the server's one pass
	$(PYTEST) -q --cov --cov-report=term-missing:skip-covered

.PHONY: floor
floor: ## The requires-python floor installs and imports (server's `floor` job)
	@if command -v $(FLOOR_PYTHON) >/dev/null 2>&1; then \
	  tmp=$$(mktemp -d) && trap 'rm -rf "$$tmp"' EXIT && \
	  $(FLOOR_PYTHON) -m venv "$$tmp/venv" && \
	  "$$tmp/venv/bin/pip" -q install . && \
	  "$$tmp/venv/bin/python" -c "import vocal_helper" && \
	  echo "floor: installs and imports on $$($(FLOOR_PYTHON) -V)"; \
	else \
	  echo "floor: no $(FLOOR_PYTHON) on PATH — syntax check at py310 only"; \
	  $(RUFF) check --target-version py310 --select E9 vocal_helper; \
	fi

.PHONY: precommit
precommit: ## pre-commit hygiene hooks over all files (whitespace, yaml, toml…)
	pre-commit run --all-files

.PHONY: preflight
preflight: env-check precommit lint format-check test floor ## Full local CI mirror — run before pushing
	@echo "✅ preflight green — the server CI gates will pass"

.PHONY: ci
ci: preflight ## Alias for preflight

.PHONY: install-hooks
install-hooks: ## Wire preflight into a git pre-push hook (safe to re-run)
	@bash scripts/install-hooks.sh
