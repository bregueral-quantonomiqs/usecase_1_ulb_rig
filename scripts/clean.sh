#!/usr/bin/env bash
set -euo pipefail

# Workspace clean script (keeps .venv and project sources)

echo "Cleaning workspace (keeping .venv and source files)..."

# Remove OS cruft
find . -name '.DS_Store' -type f -delete || true

# Remove Python and tool caches
find . -type d -name '__pycache__' -prune -exec rm -rf {} + || true
rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis || true
find . -type d -name '.ipynb_checkpoints' -prune -exec rm -rf {} + || true

# Clean results (keep top-level results dir)
if [[ -d results ]]; then
  find results -mindepth 1 -maxdepth 1 -exec rm -rf {} + || true
fi

# Remove empty quantum/results dir if present
rmdir quantum/results 2>/dev/null || true

echo "Done. Summary:" && ls -la && echo "\nresults dir:" && ls -la results || true

