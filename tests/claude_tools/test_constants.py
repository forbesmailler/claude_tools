"""Tests for claude_tools.constants."""

from claude_tools.constants import (
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
    load_config,
)


def test_repos_dir_is_absolute():
    assert REPOS_DIR == r"C:\Users\forbe\repos"


def test_mamba_activate_path():
    assert MAMBA_ACTIVATE == r"C:\Users\forbe\miniforge3\Scripts\activate.bat"


def test_mamba_bat_path():
    assert MAMBA_BAT == r"C:\Users\forbe\.local\share\mamba\condabin\mamba.bat"


def test_gh_owner():
    assert GH_OWNER == "forbesmailler"


def test_gh_env_name():
    assert GH_ENV_NAME == "setup"


def test_repo_visibility():
    assert REPO_VISIBILITY == "private"


def test_conda_channel():
    assert CONDA_CHANNEL == "conda-forge"


def test_default_deps():
    assert DEFAULT_DEPS == ["python", "invoke", "ruff", "pytest", "pytest-cov"]


def test_line_length():
    assert LINE_LENGTH == 88


def test_year_is_current():
    from datetime import datetime

    assert YEAR == datetime.now().year


def test_gitignore_contains_key_patterns():
    assert "__pycache__/" in GITIGNORE
    assert "*.py[codz]" in GITIGNORE
    assert ".vscode/" in GITIGNORE
    assert ".ruff_cache/" in GITIGNORE
    assert ".env" in GITIGNORE
    assert GITIGNORE.count("\n") == 166


def test_gitignore_ends_with_newline():
    assert GITIGNORE.endswith("\n")


def test_license_template_has_placeholders():
    assert "{year}" in LICENSE_TEMPLATE
    assert "{owner}" in LICENSE_TEMPLATE


def test_license_template_formatting():
    result = LICENSE_TEMPLATE.format(year=2025, owner="TestOwner")
    assert "2025" in result
    assert "TestOwner" in result
    assert "{year}" not in result
    assert "{owner}" not in result


def test_license_template_starts_with_mit():
    assert LICENSE_TEMPLATE.startswith("MIT License")


def test_claude_settings_is_valid_json():
    import json

    data = json.loads(CLAUDE_SETTINGS)
    assert data == {"permissions": {"defaultMode": "bypassPermissions"}}


def test_load_config_returns_expected_sections():
    cfg = load_config()
    assert "paths" in cfg
    assert "github" in cfg
    assert "setup" in cfg
    assert "iterate" in cfg


def test_load_config_paths_match_module_constants():
    cfg = load_config()
    assert cfg["paths"]["repos_dir"] == REPOS_DIR
    assert cfg["paths"]["mamba_activate"] == MAMBA_ACTIVATE
    assert cfg["paths"]["mamba_bat"] == MAMBA_BAT


def test_load_config_github_owner_matches():
    cfg = load_config()
    assert cfg["github"]["owner"] == GH_OWNER


def test_load_config_custom_path(tmp_path):
    custom = tmp_path / "test_config.yaml"
    custom.write_text("paths:\n  repos_dir: /tmp/test\n")
    cfg = load_config(custom)
    assert cfg["paths"]["repos_dir"] == "/tmp/test"
