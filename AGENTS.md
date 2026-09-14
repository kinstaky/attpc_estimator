# ATTPC estimator

## Project/Package manager
+ Prefer uv in python project (always use uv if possible).
+ Prefer pnpm in webui project.

## Implementation stack preference

### Python
+ Prefer fastAPI+uvicorn+pydantic for server.

### Frontend
+ Prefer TypeScript+vue+vuetify.

## Project

- Python package code lives in `src/attpc_estimator`; run CLIs with `uv run <command>`.
- Frontend code lives in `frontend`; use `pnpm --dir frontend run build` and `pnpm --dir frontend run typecheck`.
- Runtime paths and per-detector processing rules come from the selected TOML config (for example `config/rcnp2025.toml`); do not hard-code workspace paths.
- GAGG raw rows include an over-range flag in column 0. Trace samples begin at column 1; preserve this distinction when adding readers or fits.
- GAGG fit constants (`p0`, `p1`, `tau`, clipping thresholds) are in `config/gagg_fit_parameters.toml`. Keep batch/pipeline behavior separate from Browse-only visual fitting experiments.
