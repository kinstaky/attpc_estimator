# Trace Label

Trace Label is a local web app for browsing detector traces and assigning labels through a keyboard-first WebUI. The backend is a FastAPI service, and the frontend is a Vue app served from `frontend/dist`.

## Requirements

- Python 3.10+
- `uv`
- Node.js and `pnpm` for building the frontend

## Setup

Install the Python dependencies:

```bash
uv sync
```

Install the frontend dependencies:

```bash
cd frontend
pnpm install
```

Build the frontend bundle that FastAPI serves:

```bash
cd frontend
pnpm build
```

## Run

Start the app from the repository root:

```bash
uv run label -w <workspace> -r <run> -d <sqlite-dir>
```

Generate a 2D histogram of FFT CDF values with:

```bash
uv run cdf -t <trace-path> -w <workspace> -r <run>
uv run cdf -t <trace-path> -w <workspace> -r <run> --labeled
```

CDF arguments:

- `-t`, `--trace-path`: trace file or directory containing `run_<run>.h5` files
- `-w`, `--workspace`: workspace directory containing outputs and `trace_label.sqlite3`
- `-r`, `--run`: required run identifier
- `--baseline-window-scale`: optional FFT baseline-removal scale, default `20.0`

The `cdf` command writes:

- full traces: `workspace/run_<run>_cdf.npy`
- labeled traces: `workspace/run_<run>_labeled_cdf.npy`

Generate peak-amplitude histograms with:

```bash
uv run amplitude -t <trace-path> -w <workspace> -r <run>
uv run amplitude -t <trace-path> -w <workspace> -r <run> --labeled
```

Generate a simple filter file for the viewer with:

```bash
uv run filter -t <trace-path> -w <workspace> -r <run> --amplitude 20 60
```

This writes `filter_run_<run>_*.npy` files with `run,event_id,trace_id` rows that the viewer app can load.

Launch the histogram and filtered-trace viewer with:

```bash
uv run viewer -t <trace-path> -w <workspace>
```

The viewer contains:

- a left navigation sidebar
- a `Histograms` page with `CDF` and `Amplitude` tabs plus `All traces` / `Labeled` selection
- a `Trace Review` page for browsing traces from backend-discovered `filter_*.npy` files

Label arguments:

- `-w`, `--workspace`: directory containing `run_<run>.h5`
- `-r`, `--run`: run identifier used to reconstruct `<workspace>/run_<run>.h5`
- `-d`, `--database-dir`: directory where `trace_label.sqlite3` is stored
- `--port`: optional preferred HTTP port, default `8765`

The app prints the selected port when it starts. If needed, open `http://127.0.0.1:<port>` manually in your browser.

## WebUI Workflow

### Welcome screen

- Click `Start` or press `Space` to begin labeling.
- Click `Add` to create a new strange-label shortcut.
- The welcome panel shows the input file, database file, progress, and the current summary counts.
- The welcome panel shows the trace source path, database file, progress, and the current summary counts.

### Label screen

- The center panel shows the current trace plot.
- In browse mode, switch between `Raw` and `Analysis` views to compare the raw trace, baseline-removed trace, and Fourier magnitude spectrum.
- The left sidebar shows normal-label bucket counts.
- The right sidebar shows strange-label counts and their shortcut keys.

### Keyboard controls

- `Space` on the welcome screen: start labeling
- `Left Arrow` or `h`: enter normal-label mode
- `Right Arrow` or `l`: enter strange-label mode
- `Up Arrow` or `k`: previous trace
- `Down Arrow` or `j`: next trace
- `f`: toggle browse visual mode between raw-only and analysis
- `q` or `Escape`: leave the current mode, or return to the welcome screen from browse mode

### Normal labels

After entering normal-label mode:

- `Space`: label as `0 peak`
- `1` to `8`: label as that peak count
- `9`: label as `9+ peaks`

### Strange labels

After entering strange-label mode:

- Press the shortcut key assigned to the strange label you want to apply.
- Strange labels are created from the `Add` dialog using a name and a single shortcut key.

## Frontend Development

Run the Vite dev server from `frontend/`:

```bash
pnpm dev
```

For production-style assets used by the backend:

```bash
pnpm build
```

## Testing

Run the Python tests with:

```bash
uv run pytest
```
