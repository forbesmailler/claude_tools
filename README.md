# claude_tools

CLI utilities for bootstrapping Python projects and running iterative Claude Code improvement loops.

## Scripts

### `setup.py` — Project scaffolding

Scaffolds a new Python repo with standard structure (source dir, tests, git, mamba env, GitHub remote, VS Code) from a single command:

```bash
python claude_tools/setup.py <project_name>
```

Creates: directory layout, `.gitignore`, MIT license, `environment.yaml`, `tasks.py`, `README.md`, `CLAUDE.md`, Claude settings, then initialises git, creates a mamba environment, pushes to GitHub, and opens VS Code.

### `iterate.py` — Iterative Claude Code task runner

Runs a series of coding tasks via `claude -p`, committing after each successful iteration and squashing commits per task. Auto-retries on rate limits with automatic wait detection. Press `q` to stop gracefully.

```bash
python claude_tools/iterate.py                                    # Run all default tasks
python claude_tools/iterate.py --model opus                       # Specify model
python claude_tools/iterate.py -p "Fix all TODOs" -p "Add tests"  # Custom prompts
python claude_tools/iterate.py -t bugs -t tests                   # Run specific default tasks
python claude_tools/iterate.py --max-iterations 5                  # Override iteration cap
```

Default tasks (selectable via `-t`): `bugs`, `tests`, `concise`, `optimize`, `config`, `markdown`.

### `setup_constants.py` — Setup configuration

Loads `setup_config.yaml` and exports paths, GitHub settings, and file templates used by `setup.py`.

## Configuration

- **`setup_config.yaml`** — file paths, GitHub owner, default dependencies, line length.
- **`iterate_config.yaml`** — iteration limits, timeouts, polling intervals, task suffixes.

## Setup

```bash
mamba env create -f environment.yaml
mamba activate claude_tools
```

## Development

```bash
invoke format   # ruff format + check (88 char lines)
invoke test     # pytest with coverage
invoke all      # both
```

## License

MIT
