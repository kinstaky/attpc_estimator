# ATTPC Estimator

ATTPC Estimator is a local analysis workspace for RCNP AT-TPC data. It combines a FastAPI and Vue WebUI for trace and pointcloud review, labels, mapping, histograms, and event Chain review with Python commands for detector processing, filtering, and GAGG fits. The repository also retains older ROOT pipelines and analysis utilities; see [Project architecture](docs/architecture.md) for their place in the workflow.

## Requirements and setup

- Python 3.12 or newer, `uv`, Node.js, and `pnpm`.
- The editable sibling packages configured in `pyproject.toml`: `attpc_storage`, `attpc_pointcloud`, `attpc_runs`, and `py_brill_rcnp2025` at their listed relative paths.

From the repository root:

```bash
uv sync --group dev
pnpm --dir frontend install
pnpm --dir frontend run build
```

The WebUI serves the built frontend through FastAPI. For frontend development, use `pnpm --dir frontend run dev` alongside the Python server and open the Vite URL.

## Configure a workspace

Copy or edit `config/rcnp2025.toml` for your installation, then pass it to each command with `-c`. Its top-level `trace_path` points to a `run_<run>.h5` file or a directory of such files; `workspace` holds analysis outputs. Set the example's machine-specific paths, including `[webui].brill_config_path`, to locations on your machine. CLI `-t`, `-w`, and `-r` override the corresponding config values where supported. Run `uv run <command> --help` for all options.

The config groups detector baseline, amplitude, time, bitflip, saturation, and GAGG channel rules under `[attpc]`, `[ic]`, `[si]`, and `[gagg]`. `[filter]`, `[pointcloud]`, and `[findline]` control selection and pointcloud analysis. `[run_tags].path` names the run workbook for the older ROOT pipelines. Paths in the run-tags and GAGG fit-parameter settings are resolved relative to the config file. Batch GAGG fit constants are in the file named by `[gagg].fit_parameters_path`, usually `config/gagg_fit_parameters.toml`.

**Older ROOT pipeline input layout:** `ic`, `silicon`, and `gagg` currently pass `hdf5/run_<run>.h5` relative to `workspace` to `attpc_storage`. Ensure those files exist in `workspace/hdf5/` when using these commands. Other readers, including the WebUI, `filter`, `pointcloud`, and `gagg-fit`, use `trace_path`. The ROOT pipelines also use the workbook from `[run_tags].path` and track execution in `workspace/runs.sqlite`.

GAGG raw rows have an over-range flag in column 0; samples start in column 1. Preserve this distinction when interpreting raw data or fits.

## Use the WebUI

```bash
uv run webui -c config/rcnp2025.toml
```

The server prints its port. Open `http://127.0.0.1:<port>`. The WebUI provides:

| Area | Use |
| --- | --- |
| Label | Assign trace and pointcloud event labels. |
| Browse | Inspect raw traces, derived pointclouds, and Chain stages. |
| Mapping | Review silicon strip and AT-TPC pad mappings. |
| Histograms | Generate and inspect raw, labeled, filtered, detector, and pointcloud histograms. |
| Settings | Change display appearance. |

Page and tab selections, the route, and the current runtime session are saved in `workspace/webui_session.json`. Labels are stored separately in `workspace/labels.db`.

Chain review follows an event through raw traces, ingot outputs, and matched D1/D2/GAGG candidates. Its ingot stage shows silicon combination guesses and distance/energy matrices. The match stage uses calibrated PID markers, histogram backgrounds, and optional CATIMA theory curves. Set `[webui].brill_config_path` and `[webui.browse.chain.match]` in the config; add theory curves with `[[webui.browse.chain.match.theory_lines]]`. PID backgrounds may take time to generate on first use. In Chain, `h`/`l` or Left/Right move one event, `H`/`L` move to an available event, `n`/`N` switch stages, and number keys cycle candidates or ingot guesses.

## Process data

The installed commands are declared in `pyproject.toml`. These examples use run 1056; choose a run present in your input data. The ROOT detector commands also require it in the run workbook.

```bash
uv run ic -c config/rcnp2025.toml -r 1056
uv run silicon -c config/rcnp2025.toml -r 1056
uv run gagg -c config/rcnp2025.toml -r 1056
uv run gagg-fit -c config/rcnp2025.toml -r 1056 --output-root
uv run pointcloud -c config/rcnp2025.toml -r 1056
uv run filter -c config/rcnp2025.toml -r 1056
uv run relabel -c config/rcnp2025.toml -r 1056 --label noise
```

| Command | Main result |
| --- | --- |
| `ic`, `silicon`, `gagg` | Detector ROOT trees under `workspace/ingot/`. |
| `gagg-fit` | `workspace/trace/gagg_fit_<run>.parquet`, plus ROOT with `--output-root`. |
| `pointcloud` | `workspace/pointcloud/run_<run>.h5` unless `--output` is set. |
| `filter` | A named selection `.npy` in `workspace/filter/`, unless `--output` is set. |
| `relabel` | A labeled selection `.npy` in `workspace/`; label modes include `noise`, `oscillation`, and `saturation`. |

`filter` applies a configured selection and defaults to a limited scan; see `[filter].limit` and `--help` before running over a full run. Histograms are generated through the WebUI service and cached under `workspace/histograms/`; the retired per-metric histogram CLIs are not installed. Shell scripts in `scripts/` and the notebook in `notebooks/` are experiment examples with their own paths and run lists.

## Check changes

```bash
uv run --group dev pytest -q
pnpm --dir frontend run typecheck
pnpm --dir frontend run build
```

See [Project architecture](docs/architecture.md) for the module map, data flow, stored files, and retained older code. Contributors and coding agents should also read [AGENTS.md](AGENTS.md).
