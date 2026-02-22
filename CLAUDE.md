# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLI utilities for bootstrapping Python projects and running iterative Claude Code improvement loops. Each script has its own config:

- **`setup.py`** — scaffolds a new Python repo (files, git, mamba env, GitHub remote, VS Code) from a single `python setup.py <name>` command. Imports templates and paths from `setup_constants.py`.
- **`iterate.py`** — runs a sequence of coding tasks via `claude -p`, committing after each iteration, auto-retrying on rate limits, and squashing commits per task. Loads `iterate_config.yaml` directly.
- **`setup_constants.py`** — loads `setup_config.yaml`, exports paths and file templates (gitignore, license, Claude settings).
- **`setup_config.yaml`** — setup parameters: paths, GitHub owner, dependencies, line length.
- **`iterate_config.yaml`** — iterate parameters: iteration limits, timeouts, polling intervals, task suffixes.

## Dependencies

Managed via `environment.yaml` (mamba/conda). Install: `mamba env create -f environment.yaml`

## Development

```bash
invoke format   # ruff format + check (88 char lines)
invoke test     # pytest with coverage
invoke all      # both
```

## Architecture Notes

- Scripts in `claude_tools/` are run directly (`python claude_tools/setup.py <name>`, `python claude_tools/iterate.py`), not as an installed package.
- `iterate.py` uses `claude -p --dangerously-skip-permissions --output-format stream-json --verbose`, parsing text_delta events to stream output to the terminal in real-time. Rate limit detection uses regex against combined stdout+stderr.
- `iterate.py` context: iterations 2+ use `--continue` to preserve conversation context. On failure or "prompt too long", falls back to a fresh session with `git diff --stat` of changes so far. No context is carried between tasks.
- `iterate.py` git workflow: commit after each iteration, squash all iteration commits into one per task via `git reset --soft`.
- `commit_changes()` excludes `logs/iterate_log.md` from staging.
- Setup paths and parameters live in `setup_config.yaml`, loaded by `setup_constants.load_config()`. Iterate parameters live in `iterate_config.yaml`, loaded inline by `iterate.py`.
- Default task prompts in `iterate.py` must be maximum 4 string lines (excluding the title). "Read CLAUDE.md" is prefixed automatically at runtime.
