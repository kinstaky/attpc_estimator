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
- Persist all WebUI page/tab state in `webui_session.json`. When adding a page or tab, add its UI-state schema, defaults, validation, hydration, and snapshot persistence.
- Chain review spans raw, ingot, and match stages in `service/chain.py` and `frontend/src/views/ChainView.vue`. It uses Brill viewers for silicon combinations and calibrated match energies, and CATIMA for optional PID theory lines. Keep native viewer calls serialized and return finite numbers or `None` in API payloads.
- CLI entry points are declared in `pyproject.toml`; check them before documenting commands. `webui` replaces the old `label` and `viewer` entry points. The WebUI config and run choices are in `config/rcnp2025.toml`.
- Keep tests aligned with entry points in `pyproject.toml`. The old per-metric histogram CLIs were retired; histogram generation now runs through the WebUI service.

## Whole-project map

- Read [README.md](README.md) for use and [docs/architecture.md](docs/architecture.md) for module and artifact ownership before changing a workflow.
- Installed commands live in `pyproject.toml`: `webui`, `ic`, `silicon`, `gagg`, `gagg-fit`, `pointcloud`, `filter`, and `relabel`. Do not assume an older standalone module or notebook is an installed CLI.
- `cli/` parses commands and TOML; `server.py` and `service/` handle WebUI requests; `process/` holds numerical operations; `pipeline/` writes detector and pointcloud outputs; `storage/` owns workspace formats and paths. Frontend routes, pages, components, and stores live under `frontend/src/`.
- The older IC, silicon, and GAGG ROOT pipelines still read `workspace/hdf5/run_<run>.h5` through `attpc_storage`, use `[run_tags].path` and `workspace/runs.sqlite`, and write `workspace/ingot/`. WebUI, filter, pointcloud, and GAGG batch fitting read the configured `trace_path`. Preserve these existing input and output contracts during cleanup.
- `estimate/silicon_rise_time.py`, `storage/labeled_traces.py`, `scripts/`, and `notebooks/` include standalone or older workflows. Inspect callers, exported APIs, formats, and documented use before removing apparently unused code.
- Labels are stored in `workspace/labels.db`; UI and runtime state in `workspace/webui_session.json`; histogram artifacts in `workspace/histograms/`; pointclouds in `workspace/pointcloud/`; GAGG batch fits in `workspace/trace/`. Use helpers in `storage/run_paths.py` where available.
- Keep API payloads, CLI options, persisted files, and numerical results stable when refactoring. For cleanup, run relevant Python tests and frontend typecheck/build, then update both human and agent documentation for changed behavior.
