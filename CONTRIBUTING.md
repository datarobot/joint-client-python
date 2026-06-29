# Contributing to `jointfm-client`

Thanks for your interest in improving the JointFM Python SDK. This repository
ships the public `jointfm-client` package on PyPI, so every change is shipped to
external users — please read this guide before opening a pull request.

## Ground Rules

- The SDK is licensed under **Apache-2.0**. By contributing, you agree that
  your contribution is licensed under the same terms (inbound = outbound).
- Follow the project rules in [AGENTS.md](AGENTS.md). They take precedence
  over anything stated here.
- Be respectful in issues, reviews, and discussions.

## Reporting Issues

- **Bugs and feature requests:** open a GitHub issue at
  [datarobot/joint-client-python](https://github.com/datarobot/joint-client-python/issues)
  with a minimal reproduction, the SDK version (`uv run python -c "import
  jointfm_client; print(jointfm_client.__version__)"`), and the Python version.
- **Security issues:** do not open a public issue. Follow the disclosure
  process in [SECURITY.md](SECURITY.md) (if present) or email the DataRobot
  security team directly.

## Local Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin
task setup
```

`task setup` creates `.venv`, installs dev dependencies, registers pre-commit
hooks (including the `commit-msg` hook), and ensures `typos` is available.

Supported development platforms: Ubuntu, AWS Linux, and macOS. Windows is not
supported for development; do not add Windows-specific code paths.

## Development Loop

| Command | Purpose |
| --- | --- |
| `task lint` | Ruff lint (read-only) |
| `task format` | Ruff format (rewrites files) |
| `task typecheck` | `ty` static type check |
| `task test` | Unit test suite |
| `task coverage` | Tests with coverage floor enforcement |
| `task check` | Read-only static gates (typos, lint, format check, typecheck) |
| `task pre-commit` | Run every pre-commit hook on all files |
| `task build` | Build the sdist + wheel and validate metadata |

You must run `task pre-commit` and fix every reported issue before reporting a
change as done. After it succeeds, review the resulting diff and explain why
each hunk is necessary in the pull request.

## Branch And Pull Request Flow

1. Fork the repository (or branch directly if you are a DataRobot maintainer).
2. Create a feature branch off `main` with a descriptive name.
3. Make focused commits that pass `task pre-commit` locally.
4. Open a pull request against `main`. Describe what changed, why, and how
   you verified it.
5. Keep pull requests small and reviewable. Refactors, formatting-only
   changes, and behavior changes should be separate commits and ideally
   separate pull requests.

## Commit Messages

Every commit subject must follow [Conventional
Commits](https://www.conventionalcommits.org/):

```text
<type>(<optional scope>): <imperative summary>
```

See the **Versioning & Commits** section of [README.md](README.md) for the
full list of accepted types, when each one triggers a SemVer bump, and how
to mark breaking changes. The `commit-msg` pre-commit hook runs `cz check`
and rejects malformed messages locally.

## Public API Surface

The SDK's public surface is everything re-exported from
[`src/jointfm_client/__init__.py`](src/jointfm_client/__init__.py). Names that
start with an underscore are implementation details.

- Adding to `__all__` is a `feat`.
- Removing from `__all__`, renaming an exported name, or changing the shape of
  a public class, function, or exception is a **breaking change**. Use the
  Conventional Commits `!` marker or a `BREAKING CHANGE:` footer.
- The wire contract pinned by `schema_version = "v1"` is owned by the JointFM
  service team. Do not change the request or response shape unilaterally.
- New public exports must also be documented in
  [docs/api-reference.md](docs/api-reference.md) in the same change.

## Tests

- Add or update tests for every behavior change. The pre-commit gate enforces a
  coverage floor; uncovered new code will fail the gate.
- Tests must be fast. The live DataRobot smoke test
  ([tests/test_live_smoke.py](tests/test_live_smoke.py)) is gated by environment
  variables and skipped by default.

## Releases

Releases are cut by maintainers using Commitizen. See the **Cutting a release**
section of [README.md](README.md). Do not hand-edit the `version =` line in
`pyproject.toml`, `__version__` in `src/jointfm_client/__init__.py`, or the
"Current SDK package version" line in `README.md` — those are owned by
`cz bump`.

## Git Safety

Never run destructive git operations unless a maintainer explicitly asks for
that specific command — no `git push --force`, no `git reset --hard`, no
history rewrites, no `--no-verify` to bypass hooks, and no discarding of
uncommitted work.
