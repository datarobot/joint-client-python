# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases are cut with `task release`, which uses Commitizen to read
[Conventional Commits](https://www.conventionalcommits.org/) since the last
release tag, infer the next SemVer bump, prepend the new section here,
commit the bump, and create an annotated tag.

## v0.1.0 (2026-07-06)

### Feat

- enhance release process with pre-bump hooks and improved checks
- add GitHub Actions workflow for publishing to PyPI and authors file
- introducing commitizen chore: introducing ruff format

### Fix

- remove default_sample_count from HealthMetadata and related tests
- add copyright notice
- no unnecessary typo exceptions
- pre-commit install type

## v0.0.1 (2026-06-26)
