#!/usr/bin/env python3
# Copyright (c) 2026 DataRobot, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Insert the project license header at the top of every source file.

Supported file types:

- Python source (``.py``): prepend a ``#``-comment header (after the
  shebang if one is present).
- Jupyter notebooks (``.ipynb``): insert a markdown cell at index 0
  containing the header as an HTML comment. The cell renders as
  whitespace for human readers but is scannable by tooling and travels
  with the notebook if it is downloaded standalone. Notebooks are
  rewritten with ``nbformat.read`` / ``nbformat.write`` so cell outputs
  are preserved unchanged.

Per-file template overrides: if ``.license-overrides.toml`` exists at the
repository root it lists alternate templates for specific files (used,
for example, by Apache-2.0 derivative files that need dual-attribution
headers). Files not matched by any override use the default template
passed via ``--license-filepath``.

Idempotent: files that already carry the correct header are skipped, so
the hook cannot double-stamp existing headers. Exits 1 (and lists the
modified files) if any file needed a header, 0 otherwise. Matches
pre-commit's expected autofix-hook behavior.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
import tomllib
from pathlib import Path

DETECT_LINES = 5
OVERRIDES_FILE = ".license-overrides.toml"
# Any SPDX-License-Identifier line counts as "notebook header present" —
# the exact text differs between default and derivative templates.
NOTEBOOK_MARKER = "SPDX-License-Identifier:"


def render_python_header(template: Path) -> tuple[str, str]:
    """Return (rendered ``#``-comment block, marker line) for ``.py`` files."""
    raw = template.read_text(encoding="utf-8").rstrip("\n").splitlines()
    if not raw:
        raise SystemExit(f"empty license template: {template}")
    rendered = "\n".join(f"# {line}".rstrip() for line in raw) + "\n\n"
    marker = f"# {raw[0]}".rstrip()
    return rendered, marker


def render_notebook_cell(template: Path) -> str:
    """Return the visible markdown source for a notebook header cell.

    Each template line becomes its own paragraph (joined by blank lines)
    so the SPDX attribution renders as readable text in the notebook UI,
    not as a hidden HTML comment.
    """
    raw = template.read_text(encoding="utf-8").rstrip("\n").splitlines()
    if not raw:
        raise SystemExit(f"empty license template: {template}")
    return "\n\n".join(raw)


def load_overrides(root: Path) -> list[tuple[list[str], Path]]:
    """Parse ``.license-overrides.toml`` into ``(patterns, template_path)`` rows."""
    path = root / OVERRIDES_FILE
    if not path.exists():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    result: list[tuple[list[str], Path]] = []
    for override in data.get("overrides", []):
        patterns = override["patterns"]
        template = root / override["template"]
        if not template.exists():
            raise SystemExit(f"override template not found: {template}")
        result.append((patterns, template))
    return result


def select_template(
    file_path: str,
    overrides: list[tuple[list[str], Path]],
    default: Path,
) -> Path:
    """Return the first override template matching ``file_path``, else default."""
    for patterns, template in overrides:
        if any(fnmatch.fnmatch(file_path, pat) for pat in patterns):
            return template
    return default


def has_py_header(content: str, marker: str) -> bool:
    """Return True if a line within DETECT_LINES starts with ``marker``.

    Uses startswith (not equality) so existing headers with trailing
    inline comments such as ``# noqa`` still count as present.
    """
    marker = marker.rstrip()
    for line in content.splitlines()[:DETECT_LINES]:
        if line.startswith(marker):
            return True
    return False


def insert_py_header(content: str, header: str) -> str:
    """Prepend the header, preserving any leading shebang line."""
    if content.startswith("#!"):
        shebang, _, rest = content.partition("\n")
        return f"{shebang}\n{header}{rest}"
    return f"{header}{content}"


def process_py(path: Path, template: Path) -> bool:
    """Insert the ``.py`` header if missing; return True if file was modified."""
    header, marker = render_python_header(template)
    content = path.read_text(encoding="utf-8")
    if has_py_header(content, marker):
        return False
    path.write_text(insert_py_header(content, header), encoding="utf-8")
    return True


def process_ipynb(path: Path, template: Path) -> bool:
    """Insert the notebook header cell if missing; return True if modified.

    Uses ``nbformat.read`` / ``nbformat.write`` to round-trip cell outputs
    unchanged — this repository deliberately commits notebook outputs.
    """
    # Lazy import so .py-only invocations don't require nbformat.
    import nbformat

    nb = nbformat.read(str(path), as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "markdown" and NOTEBOOK_MARKER in cell.source:
            return False
    nb.cells.insert(
        0,
        nbformat.v4.new_markdown_cell(
            render_notebook_cell(template),
            metadata={"id": "license-header", "language": "markdown"},
        ),
    )
    nbformat.write(nb, str(path))
    return True


def process_file(path: Path, template: Path) -> bool:
    """Dispatch to the per-extension handler. Return True if file modified."""
    if path.suffix == ".py":
        return process_py(path, template)
    if path.suffix == ".ipynb":
        return process_ipynb(path, template)
    return False


def main(argv: list[str] | None = None) -> int:
    """Insert the configured header into any file that does not have it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--license-filepath", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)

    default_template = args.license_filepath
    overrides = load_overrides(args.repo_root)
    repo_root = args.repo_root.resolve()

    modified = 0
    for path in args.files:
        try:
            rel = path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            rel = str(path)
        template = select_template(rel, overrides, default_template)
        if process_file(path, template):
            print(f"inserted header: {rel} (template: {template.name})")
            modified += 1

    if modified:
        print(f"insert-license: modified {modified} file(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
