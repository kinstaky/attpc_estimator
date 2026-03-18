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
uv run label -i <input-file> -d <sqlite-dir>
```

Generate per-trace FFT CDF samples with:

```bash
uv run batch -i <input-file>
```

Arguments:

- `-i`, `--input-file`: input HDF5 trace file
- `-d`, `--database-dir`: directory where `trace_label.sqlite3` is stored
- `--port`: optional preferred HTTP port, default `8765`

Batch arguments:

- `-i`, `--input-file`: input HDF5 trace file
- `-o`, `--output-file`: optional output `.npy` file
- `--baseline-window-scale`: optional FFT baseline-removal scale, default `20.0`

The batch command writes a NumPy array with shape `(num_traces, 10)` containing:
`F(10), F(20), F(30), F(40), F(50), F(60), F(100), F(150), F(200), F(250)`.

The app prints the selected port when it starts. If needed, open `http://127.0.0.1:<port>` manually in your browser.

## WebUI Workflow

### Welcome screen

- Click `Start` or press `Space` to begin labeling.
- Click `Add` to create a new strange-label shortcut.
- The welcome panel shows the input file, database file, progress, and the current summary counts.

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
