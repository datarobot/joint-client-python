# Copyright (c) 2026 DataRobot, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the notebooks surface of jointfm_client."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

from jointfm_client import bootstrap_notebook, resolve_notebook_project_root


def _make_workspace(tmp_path: Path) -> tuple[Path, Path]:
    """Make workspace."""
    repo_root = tmp_path / "joint-client-python"
    repo_root.mkdir()
    (repo_root / "Taskfile.yaml").write_text("version: '3'\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'jointfm-client-test'\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "jointfm_client").mkdir(parents=True)
    notebook_dir = repo_root / "notebooks"
    notebook_dir.mkdir(parents=True)
    return repo_root, notebook_dir


def test_resolve_notebook_project_root_from_nested_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve notebook project root from nested dir."""
    repo_root, notebook_dir = _make_workspace(tmp_path)

    monkeypatch.chdir(notebook_dir)

    assert resolve_notebook_project_root() == repo_root


def test_resolve_notebook_project_root_from_foreign_src_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve notebook project root from foreign src layout."""
    repo_root = tmp_path / "instant-portfolio-optimization"
    notebook_dir = repo_root / "notebooks"

    (repo_root / "src" / "ipo").mkdir(parents=True)
    notebook_dir.mkdir()
    (repo_root / "Taskfile.yml").write_text("version: '3'\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'instant-portfolio-optimization'\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(notebook_dir)

    assert resolve_notebook_project_root() == repo_root


def test_bootstrap_notebook_changes_dir_and_adds_src_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bootstrap notebook changes dir and adds src root."""
    repo_root, notebook_dir = _make_workspace(tmp_path)
    src_root = repo_root / "src"
    original_sys_path = list(sys.path)

    monkeypatch.chdir(notebook_dir)

    try:
        resolved_root = bootstrap_notebook(add_src_root=True)

        assert resolved_root == repo_root
        assert Path.cwd() == repo_root
        assert sys.path[0] == str(src_root)

        bootstrap_notebook(add_src_root=True)

        assert sys.path.count(str(src_root)) == 1
    finally:
        sys.path[:] = original_sys_path


def test_resolve_notebook_project_root_raises_outside_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve notebook project root raises outside workspace."""
    with tempfile.TemporaryDirectory() as temp_dir:
        outside_dir = Path(temp_dir).resolve()

        monkeypatch.chdir(outside_dir)

        with pytest.raises(FileNotFoundError, match="Could not find a workspace root"):
            resolve_notebook_project_root()


def test_example_notebooks_start_with_bootstrap_cell() -> None:
    """The first code cell of every example notebook is the bootstrap snippet.

    Notebooks may carry a leading markdown cell (e.g. the SPDX license
    header inserted by the insert-license pre-commit hook), so the
    invariant we check is on the first *code* cell rather than cell 0.
    """
    repo_root = Path(__file__).resolve().parents[1]
    notebook_paths = sorted((repo_root / "notebooks").glob("*.ipynb"))
    expected_bootstrap_source = (
        "from jointfm_client import bootstrap_notebook\n"
        "bootstrap_notebook(add_src_root=True)"
    )

    assert [path.name for path in notebook_paths] == [
        "forecast_csv.ipynb",
        "forecast_mean.ipynb",
        "forecast_quantiles.ipynb",
        "forecast_samples.ipynb",
        "forecast_trading.ipynb",
        "pandas_result_conversion.ipynb",
        "predict_json.ipynb",
        "service_health.ipynb",
    ]
    for notebook_path in notebook_paths:
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
        assert payload["metadata"]["kernelspec"]["language"] == "python"
        first_code_cell = next(
            cell for cell in payload["cells"] if cell["cell_type"] == "code"
        )
        assert first_code_cell["metadata"]["language"] == "python"
        first_source = "\n".join(
            line.rstrip("\n") for line in first_code_cell["source"]
        )
        assert first_source == expected_bootstrap_source
        full_source = "\n".join("".join(cell["source"]) for cell in payload["cells"])
        assert "secret-token" not in full_source
        assert "DATAROBOT_API_TOKEN=" not in full_source
        if notebook_path.name == "forecast_csv.ipynb":
            assert "sys.executable" in full_source
            assert "jointfm_client.cli" in full_source
            assert "'jointfm-client'" not in full_source
        for cell in payload["cells"]:
            assert "language" in cell["metadata"]
            assert cell["metadata"]["id"]
