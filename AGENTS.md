# Repository Agent Guide

## Scope

This repository contains independently installable Python distributions for patient, hospital, robot, and simulation-ready digital twins. The repository root is also an aggregate distribution containing all four modules.

## Development

- Use Python 3.10 or newer.
- Keep distribution names kebab-cased and import package names snake-cased.
- Keep each top-level module installable on its own with `uv pip install ./<module>` or `pip install ./<module>`.
- Run `uv run --extra dev pytest` from the repository root for the aggregate package, or run `uv run --extra dev pytest` from an individual module directory.
- Add SPDX headers to Python source files.

## Shared Skills

Repository-specific skills belong in `skills/`. The `.claude/skills` and `.codex/skills` paths both reference that shared directory.
