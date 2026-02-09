# claude_tools

## Setup

```bash
mamba env create -f environment.yaml
mamba activate claude_tools
```

## Development

```bash
invoke format   # ruff format + check
invoke test     # pytest with coverage
invoke all      # both
```
