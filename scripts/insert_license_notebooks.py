#!/usr/bin/env python3
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

"""Insert/refresh the project license header cell in Jupyter notebooks.

skywalking-eyes (the ``insert-license`` hook) stamps license headers into
source files but cannot process ``.ipynb`` notebooks. This companion hook
covers notebooks: it ensures the first cell is a markdown cell carrying the
same license text, deriving that text from ``.licenserc.yaml`` so there is a
single source of truth shared with skywalking-eyes.

Cell outputs are preserved via ``nbformat.read`` / ``nbformat.write`` (some
repositories deliberately commit executed notebook outputs).

Idempotent: exits 1 (and lists the modified files) only when a notebook
needed its header inserted or refreshed, 0 otherwise. Matches pre-commit's
expected autofix-hook behavior.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

LICENSERC = ".licenserc.yaml"
CELL_ID = "license-header"
# Any SPDX identifier line marks an existing (possibly stale) header cell.
MARKER = "SPDX-License-Identifier:"


def load_header_text(repo_root: Path) -> str:
    """Return the default license header content from ``.licenserc.yaml``.

    ``header`` may be a single mapping or a list of per-path mappings (used
    when some files carry a derivative header). The notebook header uses the
    general entry: the one whose ``paths`` include ``**/*.py``, else the last.
    """
    data = yaml.safe_load((repo_root / LICENSERC).read_text(encoding="utf-8"))
    header = data["header"]
    entries = header if isinstance(header, list) else [header]
    chosen = next(
        (entry for entry in entries if "**/*.py" in entry.get("paths", [])),
        entries[-1],
    )
    return str(chosen["license"]["content"]).rstrip("\n")


def cell_source(cell: object) -> str:
    """Return a notebook cell's source as a single string."""
    source = cell.source
    return "".join(source) if isinstance(source, list) else source


def process_notebook(path: Path, header_text: str) -> bool:
    """Insert or refresh the license cell. Return True if the file changed."""
    import nbformat

    nb = nbformat.read(str(path), as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        is_header = cell.get("metadata", {}).get("id") == CELL_ID
        if is_header or MARKER in cell_source(cell):
            if cell_source(cell) == header_text:
                return False
            cell.source = header_text
            nbformat.write(nb, str(path))
            return True
    nb.cells.insert(
        0,
        nbformat.v4.new_markdown_cell(
            header_text, metadata={"id": CELL_ID, "language": "markdown"}
        ),
    )
    nbformat.write(nb, str(path))
    return True


def main(argv: list[str] | None = None) -> int:
    """Refresh the license cell in every notebook passed on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)

    header_text = load_header_text(args.repo_root)

    modified = 0
    for path in args.files:
        if path.suffix != ".ipynb":
            continue
        if process_notebook(path, header_text):
            print(f"updated notebook header: {path}")
            modified += 1

    if modified:
        print(f"insert-license-notebooks: modified {modified} file(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
