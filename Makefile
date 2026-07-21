# Local development gates — a one-command mirror of the server CI.
#
# The point: run `make preflight` (or wire it as a git hook with
# `make install-hooks`) and a green result guarantees the GitHub
# Actions pipeline in .github/workflows/ci.yml stays green. Failures
# surface here, on your machine, instead of blocking on the server.
#
# Keep these targets in lock-step with ci.yml — every hard gate the
# server enforces has a target here, and `preflight` runs the union.

RUFF   ?= ruff
PYTEST ?= pytest

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: lint
lint: ## ruff check — hard gate, mirrors the CI "Lint" job
	$(RUFF) check .

.PHONY: format-check
format-check: ## ruff format --check — informational (matches CI continue-on-error)
	-$(RUFF) format --check .

.PHONY: test
test: ## pytest with integration deselected — mirrors the CI "Tests" job
	$(PYTEST) -q

.PHONY: precommit
precommit: ## pre-commit hygiene + ruff hooks over all files — mirrors the CI "pre-commit" job
	pre-commit run --all-files

.PHONY: preflight
preflight: precommit lint format-check test ## Full local CI mirror — run before pushing
	@echo "✅ preflight green — the server CI gates will pass"

.PHONY: ci
ci: preflight ## Alias for preflight

.PHONY: install-hooks
install-hooks: ## Wire preflight into a git pre-push hook (safe to re-run)
	@bash scripts/install-hooks.sh
