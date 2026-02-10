"""Initialize a new Python project repository with standard structure."""

import os
import subprocess
import sys

from constants import (
    CLAUDE_SETTINGS,
    CONDA_CHANNEL,
    DEFAULT_DEPS,
    GH_ENV_NAME,
    GH_OWNER,
    GITIGNORE,
    LICENSE_TEMPLATE,
    LINE_LENGTH,
    MAMBA_ACTIVATE,
    MAMBA_BAT,
    REPO_VISIBILITY,
    REPOS_DIR,
    YEAR,
)


def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write(content)


def mamba_run(command):
    subprocess.run(f'call "{MAMBA_ACTIVATE}" && {command}', shell=True, check=True)


def create_files(project_dir, name):
    os.makedirs(os.path.join(project_dir, name))
    os.makedirs(os.path.join(project_dir, "tests"))
    os.makedirs(os.path.join(project_dir, ".claude"))

    for d in [name, "tests"]:
        write_file(os.path.join(project_dir, d, "__init__.py"), "")

    write_file(os.path.join(project_dir, ".gitignore"), GITIGNORE)

    write_file(
        os.path.join(project_dir, "LICENSE"),
        LICENSE_TEMPLATE.format(year=YEAR, owner=GH_OWNER),
    )

    deps_lines = "".join(f"  - {d}\n" for d in DEFAULT_DEPS)
    write_file(
        os.path.join(project_dir, "environment.yaml"),
        f"name: {name}\nchannels:\n  - {CONDA_CHANNEL}\ndependencies:\n{deps_lines}",
    )

    write_file(
        os.path.join(project_dir, "tasks.py"),
        f'''\
from invoke import task


@task
def format(c):
    """Format code with ruff."""
    c.run("ruff format --line-length {LINE_LENGTH} .")
    c.run("ruff check --line-length {LINE_LENGTH} --fix --unsafe-fixes .")


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

    dev_section = (
        "## Development\n\n```bash\n"
        "invoke format   # ruff format + check\n"
        "invoke test     # pytest with coverage\n"
        "invoke all      # both\n```\n"
    )

    write_file(
        os.path.join(project_dir, "README.md"),
        f"# {name}\n\n## Setup\n\n```bash\n"
        f"mamba env create -f environment.yaml\nmamba activate {name}\n```\n\n"
        + dev_section,
    )

    write_file(
        os.path.join(project_dir, "CLAUDE.md"),
        f"# CLAUDE.md\n\n"
        f"This file provides guidance to Claude Code (claude.ai/code) "
        f"when working with code in this repository.\n\n"
        f"## Project Overview\n\n{name}\n\n"
        f"## Dependencies\n\n"
        f"Managed via `environment.yaml` (mamba/conda).\n\n" + dev_section,
    )

    write_file(
        os.path.join(project_dir, ".claude", "settings.json"),
        CLAUDE_SETTINGS,
    )


def init_git(project_dir):
    for cmd in [
        ["git", "init"],
        ["git", "add", "."],
        ["git", "commit", "-m", "Initial commit"],
    ]:
        subprocess.run(cmd, cwd=project_dir, check=True)


def create_mamba_env(project_dir, name):
    mamba_run(f"mamba create -n {name} {' '.join(DEFAULT_DEPS)} -y")
    env_file = os.path.join(project_dir, "environment.yaml")
    mamba_run(f'mamba env export -n {name} --no-builds > "{env_file}"')


def create_github_repo(project_dir, name):
    subprocess.run(
        [
            MAMBA_BAT,
            "run",
            "-n",
            GH_ENV_NAME,
            "gh",
            "repo",
            "create",
            name,
            f"--{REPO_VISIBILITY}",
            "--source",
            ".",
            "--push",
        ],
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
