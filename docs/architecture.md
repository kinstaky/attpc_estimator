# Project architecture

This document maps the current WebUI and commands alongside the older analysis code retained in the repository. [README.md](../README.md) covers installation and use; `pyproject.toml` is the source of truth for installed CLI names.

## Data flow

1. A selected TOML file supplies the raw trace location, workspace, run, detector settings, and optional WebUI and Chain settings.
2. The WebUI reads raw `run_<run>.h5` traces through `service/traces/` and serves JSON through `server.py`. Vue pages and stores under `frontend/src/` render the results and manage navigation.
3. Labeling writes trace and pointcloud labels to `workspace/labels.db`. The WebUI writes route, page state, and runtime session to `workspace/webui_session.json`.
4. Python commands build detector ROOT trees, pointcloud HDF5 files, GAGG fit tables, and selection arrays. Histogram requests use those inputs and cached histogram artifacts.
5. Chain review combines raw traces, ingot ROOT rows, Brill matching data, and optional theory curves. It can read previously produced detector outputs.

## Python module map

| Location | Responsibility |
| --- | --- |
| `cli/` | Installed command wrappers, TOML parsing, and progress adapters. `detector_args.py` shares IC and silicon option parsing. |
| `server.py` | FastAPI app, `/api` routes, and built frontend serving. |
| `service/` | WebUI orchestration: traces, labels, pointclouds, Chain, GAGG display fitting, histogram generation and jobs. `estimator.py` owns application state and coordinates services. |
| `process/` | Numerical and batch operations: trace metrics and filters, GAGG pulse fitting, pointcloud line analysis, and histogram calculations. |
| `pipeline/` | Event processing and output for IC, silicon, GAGG, and pointcloud commands. `progress_reporter.py` supports the older detector pipelines. |
| `storage/` | Workspace paths, SQLite labels, WebUI session JSON, pointcloud compatibility, and retained labeled-trace readers. |
| `detector/`, `model/`, `utils/` | Detector maps, shared data types, trace and GAGG handling helpers. |
| `estimate/silicon_rise_time.py` | Retained standalone silicon rise-time analysis entry point; it is not an installed `pyproject.toml` script. |

The `process/` and `pipeline/` modules are used by both current commands and older workflows. `storage/labeled_traces.py` remains a compatibility reader for labeled traces. A module may have standalone use even when no installed CLI points directly at it.

## Frontend map

`frontend/src/router.ts` defines Home, Label Trace, Label Pointcloud, Browse Trace, Browse Pointcloud, Browse Chain, Mapping, Histograms, and Settings routes. `views/` owns each page; `components/` contains plots and controls; `stores/` manages backend requests and shared state. `stores/ui_state.ts` hydrates and persists the versioned WebUI state. The backend validates and saves it through `storage/webui_state.py` and `service/estimator.py`.

The frontend uses Vue, TypeScript, Vuetify, and Plotly. FastAPI serves the built bundle from `frontend/dist`; Vite serves the development frontend.

## Inputs and workspace artifacts

| Location | Meaning |
| --- | --- |
| `trace_path/run_<run>.h5` | Raw input for the WebUI, filters, pointcloud conversion, and GAGG batch fitting. `trace_path` may instead be one run file. |
| `workspace/hdf5/run_<run>.h5` | Raw input path expected by the older IC, silicon, and GAGG ROOT pipelines. |
| `workspace/ingot/` | ROOT trees from those pipelines, including IC, T0 D1/D2 silicon, and GAGG files. |
| `workspace/trace/gagg_fit_<run>.parquet` | Batch GAGG pulse-fit output; optional matching ROOT file. |
| `workspace/pointcloud/run_<run>.h5` | Derived pointcloud events, unless an explicit output path was requested. |
| `workspace/filter/*.npy` | Saved trace selections used by browsing and histogram workflows. |
| `workspace/histograms/` | Generated and cached histogram artifacts. |
| `workspace/labels.db` | SQLite trace and pointcloud labels and custom strange-label definitions. |
| `workspace/webui_session.json` | Versioned UI and runtime session state. |
| `workspace/runs.sqlite` | Run eligibility and pipeline execution database used by detector ROOT pipelines. |

Run names and some detector ROOT filenames are resolved through the external storage package; use the selected config and the command output when locating actual files. The example TOML includes site-specific absolute paths and should be adapted before use. `[run_tags].path` and `[gagg].fit_parameters_path` are resolved relative to that TOML file.

## Older workflows and compatibility

The ROOT detector pipelines still use the `attpc_storage` workspace layout and the run workbook from `attpc_runs`. The current `ic`, `silicon`, and `gagg` commands call them. Shell scripts in `scripts/` show experiment-specific parallel runs and are examples to edit before use. `notebooks/phase1/examine_pointcloud.ipynb` is an exploratory notebook. `estimate/silicon_rise_time.py` and `storage/labeled_traces.py` support older or standalone use; they are not substitutes for the installed WebUI and batch commands.

Older `label` and `viewer` commands and per-metric histogram CLI commands are not listed in `pyproject.toml`. Use `webui` and its Histograms page for those workflows. Check `pyproject.toml` when updating documentation or tests for command availability.

## Behavior boundaries

- Keep GAGG raw column 0 as the over-range flag; samples start at column 1.
- Keep configured GAGG batch-fit constants separate from Browse-only visual fitting experiments.
- Keep detector and path settings driven by the selected TOML, while preserving the older workspace layout where a pipeline still depends on it.
- When adding a WebUI page or tab, extend the UI-state schema, defaults, validation, hydration, and persistence together.
- Preserve public data formats, CLI options, API payloads, and existing analysis results during cleanup.
