"""Tests for claude_tools.setup."""

import os
import subprocess
from unittest.mock import patch

import pytest

from claude_tools.setup import (
    create_files,
    create_github_repo,
    create_mamba_env,
    init_git,
    main,
    mamba_run,
    open_vscode,
    write_file,
)


class TestWriteFile:
    def test_creates_file_and_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "file.txt"
        write_file(str(path), "hello")
        assert path.read_text() == "hello"

    def test_uses_unix_line_endings(self, tmp_path):
        path = tmp_path / "file.txt"
        write_file(str(path), "line1\nline2\n")
        raw = path.read_bytes()
        assert b"\r\n" not in raw
        assert raw == b"line1\nline2\n"

    def test_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "file.txt"
        write_file(str(path), "old")
        write_file(str(path), "new")
        assert path.read_text() == "new"

    def test_empty_content(self, tmp_path):
        path = tmp_path / "file.txt"
        write_file(str(path), "")
        assert path.read_text() == ""

    def test_existing_parent_dir(self, tmp_path):
        path = tmp_path / "file.txt"
        write_file(str(path), "content")
        assert path.read_text() == "content"


class TestMambaRun:
    @patch("claude_tools.setup.subprocess.run")
    @patch("claude_tools.setup.MAMBA_ACTIVATE", r"C:\activate.bat")
    def test_calls_subprocess_with_activate(self, mock_run):
        mamba_run("mamba create -n test python -y")
        mock_run.assert_called_once_with(
            'call "C:\\activate.bat" && mamba create -n test python -y',
            shell=True,
            check=True,
        )

    @patch(
        "claude_tools.setup.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "cmd"),
    )
    @patch("claude_tools.setup.MAMBA_ACTIVATE", r"C:\activate.bat")
    def test_raises_on_failure(self, mock_run):
        with pytest.raises(subprocess.CalledProcessError):
            mamba_run("mamba fail")


class TestCreateFiles:
    def test_creates_directory_structure(self, tmp_path):
        project_dir = str(tmp_path / "myproject")
        os.makedirs(project_dir)
        create_files(project_dir, "myproject")

        assert os.path.isdir(os.path.join(project_dir, "myproject"))
        assert os.path.isdir(os.path.join(project_dir, "tests"))
        assert os.path.isdir(os.path.join(project_dir, ".claude"))

    def test_creates_init_files(self, tmp_path):
        project_dir = str(tmp_path / "myproject")
        os.makedirs(project_dir)
        create_files(project_dir, "myproject")

        for d in ["myproject", "tests"]:
            init = os.path.join(project_dir, d, "__init__.py")
            assert os.path.isfile(init)
            with open(init) as f:
                assert f.read() == ""

    def test_creates_gitignore(self, tmp_path):
        project_dir = str(tmp_path / "proj")
        os.makedirs(project_dir)
        create_files(project_dir, "proj")

        gitignore = os.path.join(project_dir, ".gitignore")
        assert os.path.isfile(gitignore)
        with open(gitignore) as f:
            content = f.read()
        assert "__pycache__/" in content

    def test_creates_license_with_year_and_owner(self, tmp_path):
        project_dir = str(tmp_path / "proj")
        os.makedirs(project_dir)
        create_files(project_dir, "proj")

        license_file = os.path.join(project_dir, "LICENSE")
        assert os.path.isfile(license_file)
        with open(license_file) as f:
            content = f.read()
        assert "MIT License" in content
        assert "{year}" not in content
        assert "{owner}" not in content

    def test_creates_environment_yaml(self, tmp_path):
        project_dir = str(tmp_path / "proj")
        os.makedirs(project_dir)
        create_files(project_dir, "proj")

        env_file = os.path.join(project_dir, "environment.yaml")
        assert os.path.isfile(env_file)
        with open(env_file) as f:
            content = f.read()
        assert "name: proj" in content
        assert "conda-forge" in content
        assert "pytest" in content

    def test_creates_tasks_py(self, tmp_path):
        project_dir = str(tmp_path / "proj")
        os.makedirs(project_dir)
        create_files(project_dir, "proj")

        tasks_file = os.path.join(project_dir, "tasks.py")
        assert os.path.isfile(tasks_file)
        with open(tasks_file) as f:
            content = f.read()
        assert "from invoke import task" in content
        assert "--cov=proj" in content

    def test_creates_readme(self, tmp_path):
        project_dir = str(tmp_path / "proj")
        os.makedirs(project_dir)
        create_files(project_dir, "proj")

        readme = os.path.join(project_dir, "README.md")
        assert os.path.isfile(readme)
        with open(readme) as f:
            content = f.read()
        assert "# proj" in content
        assert "mamba activate proj" in content

    def test_creates_claude_md(self, tmp_path):
        project_dir = str(tmp_path / "proj")
        os.makedirs(project_dir)
        create_files(project_dir, "proj")

        claude_md = os.path.join(project_dir, "CLAUDE.md")
        assert os.path.isfile(claude_md)
        with open(claude_md) as f:
            content = f.read()
        assert "proj" in content

    def test_creates_claude_settings(self, tmp_path):
        project_dir = str(tmp_path / "proj")
        os.makedirs(project_dir)
        create_files(project_dir, "proj")

        settings = os.path.join(project_dir, ".claude", "settings.json")
        assert os.path.isfile(settings)
        import json

        with open(settings) as f:
            data = json.load(f)
        assert data["permissions"]["defaultMode"] == "bypassPermissions"

    def test_total_file_count(self, tmp_path):
        project_dir = str(tmp_path / "proj")
        os.makedirs(project_dir)
        create_files(project_dir, "proj")

        files = [
            os.path.join(root, fn)
            for root, _, filenames in os.walk(project_dir)
            for fn in filenames
        ]
        assert len(files) == 9


class TestInitGit:
    @patch("claude_tools.setup.subprocess.run")
    def test_calls_git_init_add_commit(self, mock_run):
        init_git("/fake/dir")
        assert mock_run.call_count == 3
        mock_run.assert_any_call(["git", "init"], cwd="/fake/dir", check=True)
        mock_run.assert_any_call(["git", "add", "."], cwd="/fake/dir", check=True)
        mock_run.assert_any_call(
            ["git", "commit", "-m", "Initial commit"],
            cwd="/fake/dir",
            check=True,
        )


class TestCreateMambaEnv:
    @patch("claude_tools.setup.mamba_run")
    def test_calls_mamba_create_and_export(self, mock_mamba):
        create_mamba_env("/fake/dir", "testproj")
        assert mock_mamba.call_count == 2
        mock_mamba.assert_any_call(
            "mamba create -n testproj python invoke ruff pytest pytest-cov -y"
        )
        env_file = os.path.join("/fake/dir", "environment.yaml")
        mock_mamba.assert_any_call(
            f'mamba env export -n testproj --no-builds > "{env_file}"'
        )


class TestCreateGithubRepo:
    @patch("claude_tools.setup.subprocess.run")
    @patch("claude_tools.setup.MAMBA_BAT", r"C:\mamba.bat")
    def test_calls_gh_repo_create(self, mock_run):
        create_github_repo("/fake/dir", "myrepo")
        mock_run.assert_called_once_with(
            [
                r"C:\mamba.bat",
                "run",
                "-n",
                "setup",
                "gh",
                "repo",
                "create",
                "myrepo",
                "--private",
                "--source",
                ".",
                "--push",
            ],
            cwd="/fake/dir",
            check=True,
        )


class TestOpenVscode:
    @patch("claude_tools.setup.subprocess.Popen")
    def test_opens_code_with_project_dir(self, mock_popen):
        open_vscode("/my/project")
        mock_popen.assert_called_once_with('code "/my/project"', shell=True)


class TestMain:
    @patch("claude_tools.setup.open_vscode")
    @patch("claude_tools.setup.create_github_repo")
    @patch("claude_tools.setup.init_git")
    @patch("claude_tools.setup.create_mamba_env")
    @patch("claude_tools.setup.create_files")
    @patch("claude_tools.setup.REPOS_DIR", "")
    def test_successful_run(
        self,
        mock_create_files,
        mock_mamba,
        mock_git,
        mock_gh,
        mock_vscode,
        tmp_path,
    ):
        project_dir = str(tmp_path / "newproj")
        with patch("claude_tools.setup.REPOS_DIR", str(tmp_path)):
            with patch("sys.argv", ["setup.py", "newproj"]):
                main()
        mock_create_files.assert_called_once_with(project_dir, "newproj")
        mock_mamba.assert_called_once_with(project_dir, "newproj")
        mock_git.assert_called_once_with(project_dir)
        mock_gh.assert_called_once_with(project_dir, "newproj")
        mock_vscode.assert_called_once_with(project_dir)

    def test_no_args_exits(self):
        with patch("sys.argv", ["setup.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_too_many_args_exits(self):
        with patch("sys.argv", ["setup.py", "a", "b"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_existing_dir_exits(self, tmp_path):
        existing = tmp_path / "exists"
        existing.mkdir()
        with patch("claude_tools.setup.REPOS_DIR", str(tmp_path)):
            with patch("sys.argv", ["setup.py", "exists"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1
