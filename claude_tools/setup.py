"""Initialize a new Python project repository with standard structure."""

import os
import subprocess
import sys

from constants import (
    CLAUDE_SETTINGS,
    GH_OWNER,
    GITIGNORE,
    LICENSE_TEMPLATE,
    MAMBA_ACTIVATE,
    MAMBA_BAT,
    REPOS_DIR,
    YEAR,
)


def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write(content)


def mamba_run(command):
    """Run a command inside an activated miniforge shell."""
    full = f'call "{MAMBA_ACTIVATE}" && {command}'
    subprocess.run(full, shell=True, check=True)


def create_files(project_dir, name):
    """Create all project files."""
    # Directory structure
    os.makedirs(os.path.join(project_dir, name))
    os.makedirs(os.path.join(project_dir, "tests"))
    os.makedirs(os.path.join(project_dir, ".claude"))

    # Empty __init__.py files
    for d in [name, "tests"]:
        write_file(os.path.join(project_dir, d, "__init__.py"), "")

    write_file(os.path.join(project_dir, ".gitignore"), GITIGNORE)

    write_file(
        os.path.join(project_dir, "LICENSE"),
        LICENSE_TEMPLATE.format(year=YEAR, owner=GH_OWNER),
    )

    write_file(
        os.path.join(project_dir, "environment.yaml"),
        f"name: {name}\nchannels:\n  - conda-forge\ndependencies:\n"
        f"  - python\n  - invoke\n  - ruff\n  - pytest\n  - pytest-cov\n",
    )

    write_file(
        os.path.join(project_dir, "tasks.py"),
        f'''\
from invoke import task


@task
def format(c):
    """Format code with ruff."""
    c.run("ruff format --line-length 88 .")
    c.run("ruff check --line-length 88 --fix --unsafe-fixes .")


@task
def test(c, cov=True):
    """Run tests."""
    cmd = "python -m pytest tests"
    if cov:
        cmd += " --cov={name} --cov-report=term-missing"
    c.run(cmd, pty=False)


@task(pre=[format, test])
def all(c):
    """Format and test."""
    pass
''',
    )

    write_file(
        os.path.join(project_dir, "README.md"),
        f"# {name}\n\n## Setup\n\n```bash\n"
        f"mamba env create -f environment.yaml\nmamba activate {name}\n```\n\n"
        f"## Development\n\n```bash\n"
        f"invoke format   # ruff format + check\n"
        f"invoke test     # pytest with coverage\n"
        f"invoke all      # both\n```\n",
    )

    write_file(
        os.path.join(project_dir, "CLAUDE.md"),
        f"# CLAUDE.md\n\n"
        f"This file provides guidance to Claude Code (claude.ai/code) "
        f"when working with code in this repository.\n\n"
        f"## Project Overview\n\n{name}\n\n"
        f"## Dependencies\n\n"
        f"Managed via `environment.yaml` (mamba/conda).\n\n"
        f"## Development\n\n```bash\n"
        f"invoke format   # ruff format + check\n"
        f"invoke test     # pytest with coverage\n"
        f"invoke all      # both\n```\n",
    )

    write_file(
        os.path.join(project_dir, ".claude", "settings.json"),
        CLAUDE_SETTINGS,
    )


def init_git(project_dir):
    subprocess.run(["git", "init"], cwd=project_dir, check=True)
    subprocess.run(["git", "add", "."], cwd=project_dir, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"], cwd=project_dir, check=True
    )


def create_mamba_env(project_dir, name):
    mamba_run(f"mamba create -n {name} python invoke ruff pytest pytest-cov -y")
    env_file = os.path.join(project_dir, "environment.yaml")
    mamba_run(f'mamba env export -n {name} --no-builds > "{env_file}"')


def create_github_repo(project_dir, name):
    subprocess.run(
        [MAMBA_BAT, "run", "-n", "setup",
         "gh", "repo", "create", name, "--private", "--source", ".", "--push"],
        cwd=project_dir,
        check=True,
    )


def open_vscode(project_dir):
    subprocess.Popen(f'code "{project_dir}"', shell=True)


def main():
    if len(sys.argv) != 2:
        print("Usage: python setup.py <project_name>")
        sys.exit(1)

    name = sys.argv[1]
    project_dir = os.path.join(REPOS_DIR, name)

    if os.path.exists(project_dir):
        print(f"Error: {project_dir} already exists")
        sys.exit(1)

    print(f"Creating project '{name}' at {project_dir}")

    create_files(project_dir, name)
    print("  Created project files")

    create_mamba_env(project_dir, name)
    print("  Created mamba environment")

    init_git(project_dir)
    print("  Initialized git repository")

    create_github_repo(project_dir, name)
    print("  Created GitHub repository")

    open_vscode(project_dir)
    print("  Opened VS Code")

    print(f"\nDone! Project '{name}' is ready.")


if __name__ == "__main__":
    main()
