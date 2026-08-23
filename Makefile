# Retention Lab. `make quickstart` is the contract: it must succeed on a
# clean CPU-only machine and it is exactly what CI runs on every pull request.

UV ?= uv
RUN := $(UV) run

.PHONY: quickstart env lint prose-lint test smoke battery-hash assets results mermaid-check clean

quickstart: env lint prose-lint test battery-hash smoke
	@echo "quickstart: OK"

env:
	$(UV) sync --frozen

lint:
	$(RUN) ruff check .

prose-lint:
	$(RUN) python scripts/prose_lint.py

test:
	$(RUN) pytest

# Progressive smoke target (ADR-0004): at M0 this runs the tiny end-to-end
# package path; M1 adds the battery slice and the freeze-hash check; M3
# upgrades it to the tiny knowledge-distillation run and it stays that way.
smoke:
	$(RUN) python -m retention_lab.smoke --config configs/tiny.yaml
	$(RUN) python -m retention_lab.battery.run --config configs/battery/battery.yaml --slice ci --toy

# Download pinned assets (weights, data) into the git-ignored assets/ tree,
# verifying every file against the sha256 manifest. Not part of quickstart:
# the tiny path and CI never need real weights.
assets:
	$(RUN) python -m retention_lab.data.assets

# Recompute the scoreboard hash and verify it against the freeze record.
battery-hash:
	$(RUN) python -m retention_lab.battery.freeze --check

# Requires node; runs as its own CI job so quickstart stays node-free.
mermaid-check:
	python3 scripts/check_mermaid.py

clean:
	rm -rf .venv .pytest_cache .ruff_cache

# Render every results table from tracker records; the only path into a table.
results:
	$(RUN) python -m retention_lab.tracker.results
