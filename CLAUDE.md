# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLI utilities for bootstrapping Python projects and running iterative Claude Code improvement loops. Two scripts share a constants module:

- **`setup.py`** — scaffolds a new Python repo (files, git, mamba env, GitHub remote, VS Code) from a single `python setup.py <name>` command. Imports templates and paths from `constants.py`.
- **`iterate.py`** — runs a sequence of coding tasks via `claude -p`, committing after each iteration, auto-retrying on rate limits, and squashing commits per task. Imports `load_config` from `constants.py`.
- **`constants.py`** — loads `config.yaml`, exports paths and file templates (gitignore, license, Claude settings).
- **`config.yaml`** — all tunable parameters: paths, GitHub owner, iteration limits, timeouts, polling intervals.

## Dependencies

Managed via `environment.yaml` (mamba/conda). Install: `mamba env create -f environment.yaml`

## Development

```bash
invoke format   # ruff format + check (88 char lines)
invoke test     # pytest with coverage
invoke all      # both
```

## Architecture Notes

- Scripts in `claude_tools/` are run directly (`python setup.py <name>`, `python iterate.py`), not as an installed package.
- `iterate.py` uses `claude -p --dangerously-skip-permissions --output-format json` and parses JSON output. Rate limit detection uses regex against combined stdout+stderr.
- `iterate.py` git workflow: commit after each iteration, squash all iteration commits into one per task via `git reset --soft`.
- `commit_changes()` excludes `logs/iterate_log.md` from staging.
- All machine-specific paths and tuning parameters live in `config.yaml`, loaded by `constants.load_config()`.
