from invoke import task

from claude_tools.setup_constants import LINE_LENGTH


@task
def format(c):
    """Format code with ruff."""
    c.run(f"ruff format --line-length {LINE_LENGTH} .")
    c.run(f"ruff check --line-length {LINE_LENGTH} --fix --unsafe-fixes .")


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
