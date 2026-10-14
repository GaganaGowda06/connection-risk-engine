# Connection Risk Engine

Calibrated UK rail arrival ranges and connection risk on open Darwin and HSP data.
Routes: BHM to EUS, BHM to MAN, AST to EUS via BHM (see config/routes.yaml).

## Commands
- Setup: `uv sync`, then `cp .env.example .env` and fill in keys
- Test: `uv run pytest`
- Lint: `uv run ruff check . && uv run ruff format --check .`
- Run: `uv run rail-risk check`

## Committing
- One-time: `uv run pre-commit install`
- Check everything: `uv run pre-commit run --all-files`
- Hooks scan staged changes with gitleaks and block `.env` and `data/raw/`. Never bypass with `--no-verify`.

## Data rules
- Never edit `data/raw/` or `.env`. A hook in `.claude/hooks/protect_paths.py` blocks it.
- `data/` is git-ignored except `data/.gitkeep`.
- Secrets live only in `.env`; settings use the `RAIL_` prefix.

## Stage plan
1. Repo skeleton, config, logging, CLI, protection hook.
2. Ingest Darwin and HSP data to data/raw.
3. Clean and build features with Spark.
4. Train LightGBM quantile models with conformal calibration.
5. Serve arrival ranges and connection risk over an API.
6. Terraform plan-only deployment, docs and evaluation.

## Working rules
- Run and test everything yourself.
- Test small before any large run.
- Say failures plainly with cause and next step.
- Label anything unrun UNTESTED.
- Keep secrets out of git and check before each commit.
- Confirm before destructive or externally visible actions.
- Keep replies short, answer first.
- Write prose with no em dashes and no decorative dividers.
