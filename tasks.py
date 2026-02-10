from invoke import task

from claude_tools.constants import load_config

_line_length = load_config()["setup"]["line_length"]


@task
def format(c):
    """Format code with ruff."""
    c.run(f"ruff format --line-length {_line_length} .")
    c.run(f"ruff check --line-length {_line_length} --fix --unsafe-fixes .")


@task
def test(c, cov=True):
    """Run tests."""
    cmd = "python -m pytest tests"
    if cov:
        cmd += " --cov=claude_tools --cov-report=term-missing"
    c.run(cmd, pty=False)


@task(pre=[format, test])
def all(c):
    """Format and test."""
    pass
