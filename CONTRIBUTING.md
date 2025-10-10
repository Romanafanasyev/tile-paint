# Contributing

Thanks for considering a contribution. This project aims to be simple to run and easy to improve. Keep changes small and focused.

## Getting started

1. Fork and clone the repo.
2. Create a virtual environment.
3. Install deps: `pip install -r requirements.txt`.
4. Install pre-commit hooks: `pip install pre-commit` then `pre-commit install`.
5. Run a quick smoke test:
   - From `src/`: `python -m painter.cli --input ../assets/input/sky.jpg --workload 1 --no-video`

## Branches and workflow

- Base branch for development: `dev/base`.
- Create feature branches from `dev/base`: `feat/<short-name>`, `fix/<short-name>`, `docs/<short-name>`.
- Keep PRs small. One logical change per PR.

## Commit messages

- Write clear, present-tense subjects. Examples:
  - chore: add MIT license and contributing guide
  - feat: gradient-based angle jitter option
  - fix: clamp ROI bounds on small masks

## Code style and hooks

- Pre-commit runs linters and formatters if configured. Run it manually with `pre-commit run --all-files` before pushing.
- Keep pure modules free of I O and side effects. Side effects live in `io` and `video` only.
- Prefer small functions and explicit data flow.

## Adding brushes or assets

- Place brushes under `assets/brushes/`. Grayscale PNG, white paints, black skips.
- See `docs/brushes.md` for tips and validation.

## Tests

- If you add tests, put them under `tests/`. Use plain pytest.
- Keep tests fast and focused. Mock I O and ffmpeg.

## Pull requests

1. Rebase on latest `origin/dev/base`:
   - `git fetch origin`
   - `git rebase origin/dev/base`
2. Push your branch and open a PR to `dev/base`.
3. The PR description should say what, why, and how to verify.
4. CI must be green.

## Release checklist (maintainers)

- Bump version to v1.0.0 (Git tag).
- Merge `dev/base` into `master` via PR.
- Create a GitHub release with short examples from `examples/minimal_run.md`.
- Make sure README and docs are up to date.
