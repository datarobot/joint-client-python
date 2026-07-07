# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Notebook bootstrap helpers for src-layout Python workspaces."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

PYPROJECT_MARKER: Final[str] = "pyproject.toml"
SRC_ROOT_MARKER: Final[str] = "src"
WORKSPACE_ROOT_MARKERS: Final[tuple[str, ...]] = (PYPROJECT_MARKER, SRC_ROOT_MARKER)


def _is_src_layout_project_root(candidate: Path) -> bool:
    return (candidate / PYPROJECT_MARKER).is_file() and (
        candidate / SRC_ROOT_MARKER
    ).is_dir()


def resolve_notebook_project_root(start_dir: str | Path | None = None) -> Path:
    """Return the src-layout project root for a notebook started inside a repo tree."""
    current_dir = (
        Path.cwd().resolve() if start_dir is None else Path(start_dir).resolve()
    )

    for candidate in (current_dir, *current_dir.parents):
        if _is_src_layout_project_root(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find a workspace root containing "
        f"{WORKSPACE_ROOT_MARKERS!r} from {current_dir}"
    )


def bootstrap_notebook(*, add_src_root: bool = False) -> Path:
    """Set notebook cwd to the project root and optionally prepend its ``src`` path."""
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


__all__ = [
    "WORKSPACE_ROOT_MARKERS",
    "bootstrap_notebook",
    "resolve_notebook_project_root",
]
