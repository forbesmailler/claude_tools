"""Tests for claude_tools.constants."""

from claude_tools.constants import (
    CLAUDE_SETTINGS,
    GH_OWNER,
    GITIGNORE,
    LICENSE_TEMPLATE,
    MAMBA_ACTIVATE,
    MAMBA_BAT,
    REPOS_DIR,
    YEAR,
)


def test_repos_dir_is_absolute():
    assert REPOS_DIR == r"C:\Users\forbe\repos"


def test_mamba_activate_path():
    assert MAMBA_ACTIVATE == r"C:\Users\forbe\miniforge3\Scripts\activate.bat"


def test_mamba_bat_path():
    assert MAMBA_BAT == r"C:\Users\forbe\.local\share\mamba\condabin\mamba.bat"


def test_gh_owner():
    assert GH_OWNER == "forbesmailler"


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
