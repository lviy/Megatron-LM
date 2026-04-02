# Repository Guidelines

## Project Structure & Module Organization
- `megatron/core/`: primary library code (models, transformer blocks, distributed, optimizer, datasets, inference, export).
- `megatron/training/`, `megatron/rl/`, `megatron/post_training/`, `megatron/legacy/`: training and specialized stacks around core.
- Top-level `pretrain_*.py` and `train_rl.py`: runnable entrypoints for common training flows.
- `tests/unit_tests/` and `tests/functional_tests/`: unit and end-to-end validation; functional cases live under `tests/functional_tests/test_cases/`.
- `examples/`: model/task recipes; `tools/`: developer utilities (including formatting).
- `docs/`: user/developer documentation.

## Build, Test, and Development Commands
- `uv pip install -e .`: editable local install.
- `uv sync --locked`: install default dependency groups (`linting`, `build`, `test`) from `uv.lock`.
- `uv run pytest tests/unit_tests`: run unit tests (project pytest options are defined in `pyproject.toml`).
- `bash tests/unit_tests/run_ci_test.sh --tag latest --environment dev --bucket 'tests/unit_tests/**/*.py'`: CI-style distributed unit-test launcher.
- `bash tools/autoformat.sh`: run formatter/lint/type checks on changed Python files in `megatron/core` and `tests/`.

## Coding Style & Naming Conventions
- Python style is enforced with Black + isort + pylint + ruff.
- Line length: `100` (`black`, `isort`, and `.flake8` are aligned).
- Black settings: skip string normalization and magic trailing comma behavior.
- Use explicit typing for new/changed code (called out in PR checklist).
- Naming: follow Python conventions (`snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants).

## Testing Guidelines
- Framework: `pytest` (`python_files = test_*.py`, `testpaths = tests`).
- Keep tests near the affected domain under `tests/unit_tests/<area>/`.
- Use existing markers correctly: `internal`, `flaky`, `flaky_in_dev`.
- Add both unit and functional coverage when behavior spans distributed/runtime flows.

## Commit & Pull Request Guidelines
- Match existing history style: short, imperative subjects, optionally scoped (examples: `ci: ...`, `Fix: ...`, `chore: ...`).
- Keep commits focused; avoid mixing refactors with behavior changes.
- Follow `.github/pull_request_template.md`: include tests, typing, docs, and run `tools/autoformat.sh`.
- Open PRs as draft first; mark ready only after conflicts are resolved and CI is passing.
- Add CI labels when needed (`Run tests`, `Run functional tests`) for broader validation.
