"""Notebook bootstrap helpers for the JointFM client workspace."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

WORKSPACE_ROOT_MARKERS: Final[tuple[str, ...]] = (
    "pyproject.toml",
    "Taskfile.yaml",
    "src/jointfm_client",
)


def resolve_notebook_project_root(start_dir: str | Path | None = None) -> Path:
    """Return the SDK workspace root for a notebook started inside the repo tree."""
    current_dir = Path.cwd().resolve() if start_dir is None else Path(start_dir).resolve()

    for candidate in (current_dir, *current_dir.parents):
        if all((candidate / marker).exists() for marker in WORKSPACE_ROOT_MARKERS):
            return candidate

    raise FileNotFoundError(
        "Could not find a workspace root containing "
        f"{WORKSPACE_ROOT_MARKERS!r} from {current_dir}"
    )


def bootstrap_notebook(*, add_src_root: bool = False) -> Path:
    """Set notebook cwd to the repo root and optionally prepend the root ``src`` path."""
    current_dir = Path.cwd().resolve()
    project_root = resolve_notebook_project_root(current_dir)

    if add_src_root:
        src_root = project_root / "src"
        if not src_root.is_dir():
            raise FileNotFoundError(f"Expected a src directory under {project_root}")

        src_root_str = str(src_root)
        if src_root_str not in sys.path:
            sys.path.insert(0, src_root_str)

    if current_dir != project_root:
        os.chdir(project_root)
        print(f"Changed working directory to: {project_root}")

    return project_root


__all__ = ["WORKSPACE_ROOT_MARKERS", "bootstrap_notebook", "resolve_notebook_project_root"]