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

"""Validate JointFM SDK source distribution and wheel artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import configparser
from email import policy
from email.parser import BytesParser
from pathlib import Path
import re
import tarfile
import tomllib
import zipfile


CONSOLE_SCRIPT_NAME = "jointfm-client"
CONSOLE_SCRIPT_TARGET = "jointfm_client.cli:main"
FORBIDDEN_SDIST_MEMBERS = (
    ".env",
    "config.yaml",
)
FORBIDDEN_SDIST_PREFIXES = (
    "private/",
    "notebooks/",
    "tests/",
)
REQUIRED_SDIST_MEMBERS = (
    ".env.sample",
    "LICENSE",
    "PKG-INFO",
    "README.md",
    "config.sample.yaml",
    "docs/api-reference.md",
    "pyproject.toml",
    "src/jointfm_client/__init__.py",
    "src/jointfm_client/py.typed",
)
REQUIRED_WHEEL_MEMBERS = (
    "jointfm_client/__init__.py",
    "jointfm_client/py.typed",
)


def main() -> None:
    """Validate built artifacts in the configured distribution directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", type=Path)
    parser.add_argument("--dist-dir", default="dist", type=Path)
    arguments = parser.parse_args()

    project_root = arguments.project_root.resolve()
    dist_dir = _resolve_dist_dir(project_root, arguments.dist_dir)
    project_metadata = _load_project_metadata(project_root / "pyproject.toml")
    distribution_name = project_metadata["name"]
    package_version = project_metadata["version"]
    archive_name = _archive_distribution_name(distribution_name)

    wheel_path = _expect_one(
        dist_dir.glob(f"{archive_name}-{package_version}-*.whl"),
        f"wheel for {distribution_name} {package_version}",
    )
    sdist_path = _expect_one(
        dist_dir.glob(f"{archive_name}-{package_version}.tar.gz"),
        f"source distribution for {distribution_name} {package_version}",
    )

    _validate_wheel(wheel_path, archive_name, project_metadata)
    _validate_sdist(sdist_path, archive_name, package_version)
    print(f"Validated {sdist_path.name} and {wheel_path.name}")


def _resolve_dist_dir(project_root: Path, dist_dir: Path) -> Path:
    if dist_dir.is_absolute():
        return dist_dir
    return project_root / dist_dir


def _load_project_metadata(pyproject_path: Path) -> dict[str, str]:
    with pyproject_path.open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    project = pyproject["project"]
    license_value = project["license"]
    if isinstance(license_value, dict):
        license_value = license_value["text"]
    return {
        "name": project["name"],
        "version": project["version"],
        "description": project["description"],
        "license": license_value,
        "requires_python": project["requires-python"],
    }


def _archive_distribution_name(distribution_name: str) -> str:
    return re.sub(r"[-_.]+", "_", distribution_name).lower()


def _expect_one(paths: Iterable[Path], description: str) -> Path:
    matches = sorted(paths)
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one {description}, found {len(matches)}")
    return matches[0]


def _validate_wheel(
    wheel_path: Path,
    archive_name: str,
    project_metadata: dict[str, str],
) -> None:
    package_version = project_metadata["version"]
    dist_info_dir = f"{archive_name}-{package_version}.dist-info"
    required_members = {
        *REQUIRED_WHEEL_MEMBERS,
        f"{dist_info_dir}/METADATA",
        f"{dist_info_dir}/RECORD",
        f"{dist_info_dir}/WHEEL",
        f"{dist_info_dir}/entry_points.txt",
    }
    with zipfile.ZipFile(wheel_path) as wheel_archive:
        wheel_members = set(wheel_archive.namelist())
        _require_members(wheel_members, required_members, wheel_path.name)
        _validate_wheel_metadata(
            wheel_archive.read(f"{dist_info_dir}/METADATA"),
            project_metadata,
        )
        _validate_wheel_entry_points(
            wheel_archive.read(f"{dist_info_dir}/entry_points.txt").decode("utf-8"),
        )
        init_module = wheel_archive.read("jointfm_client/__init__.py").decode("utf-8")
        _require(
            f'__version__ = "{package_version}"' in init_module,
            "wheel __version__ does not match project.version",
        )
        typed_marker = wheel_archive.read("jointfm_client/py.typed").decode("utf-8")
        _require(typed_marker.strip() != "", "wheel py.typed marker must not be empty")


def _validate_wheel_metadata(
    metadata_bytes: bytes,
    project_metadata: dict[str, str],
) -> None:
    metadata = BytesParser(policy=policy.default).parsebytes(metadata_bytes)
    _require(
        metadata["Name"] == project_metadata["name"], "wheel metadata name mismatch"
    )
    _require(
        metadata["Version"] == project_metadata["version"],
        "wheel metadata version mismatch",
    )
    _require(
        metadata["Summary"] == project_metadata["description"],
        "wheel metadata summary mismatch",
    )
    _require(
        metadata["Requires-Python"] == project_metadata["requires_python"],
        "wheel metadata Python requirement mismatch",
    )
    license_value = metadata.get("License-Expression") or metadata.get("License")
    _require(
        license_value == project_metadata["license"], "wheel metadata license mismatch"
    )
    classifiers = metadata.get_all("Classifier", [])
    _require(
        "Typing :: Typed" in classifiers, "wheel metadata must advertise typed package"
    )


def _validate_wheel_entry_points(entry_points_text: str) -> None:
    entry_points = configparser.ConfigParser()
    entry_points.read_string(entry_points_text)
    _require(
        entry_points.get("console_scripts", CONSOLE_SCRIPT_NAME)
        == CONSOLE_SCRIPT_TARGET,
        "wheel console script entry point mismatch",
    )


def _validate_sdist(sdist_path: Path, archive_name: str, package_version: str) -> None:
    root_prefix = f"{archive_name}-{package_version}/"
    required_members = {f"{root_prefix}{member}" for member in REQUIRED_SDIST_MEMBERS}
    with tarfile.open(sdist_path, "r:gz") as sdist_archive:
        sdist_members = set(sdist_archive.getnames())
    _require_members(sdist_members, required_members, sdist_path.name)
    for member in sdist_members:
        _require(
            member == root_prefix.rstrip("/") or member.startswith(root_prefix),
            f"sdist member outside archive root: {member}",
        )
        relative_member = member.removeprefix(root_prefix)
        _require(
            relative_member not in FORBIDDEN_SDIST_MEMBERS,
            f"sdist includes forbidden member: {relative_member}",
        )
        _require(
            not any(
                relative_member.startswith(prefix)
                for prefix in FORBIDDEN_SDIST_PREFIXES
            ),
            f"sdist includes forbidden member: {relative_member}",
        )


def _require_members(
    found_members: set[str],
    required_members: set[str],
    artifact_name: str,
) -> None:
    missing_members = sorted(required_members - found_members)
    _require(
        not missing_members,
        f"{artifact_name} is missing required members: {', '.join(missing_members)}",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


if __name__ == "__main__":
    main()
