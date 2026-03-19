from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from .app_factory import create_app
from .services import TraceLabelService


def main() -> None:
    args = _parse_args()
    input_path = (Path(args.workspace).expanduser().resolve() / f"run_{args.run}.h5").resolve()
    db_dir = Path(args.database_dir).expanduser().resolve()

    if not input_path.is_file():
        raise SystemExit(f"input file not found: {input_path}")
    db_dir.mkdir(parents=True, exist_ok=True)

    service = TraceLabelService(input_path=input_path, db_dir=db_dir)
    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    app = create_app(service, frontend_dist)

    port = _pick_port(args.port)
    url = f"http://0.0.0.0:{port}"
    print(f"Trace Label running at {url}")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the trace labeling app")
    parser.add_argument(
        "-w",
        "--workspace",
        required=True,
        help="Workspace directory containing run_<run>.h5 files",
    )
    parser.add_argument(
        "-r",
        "--run",
        required=True,
        type=_parse_run,
        help="Run identifier used to reconstruct <workspace>/run_<run>.h5",
    )
    parser.add_argument(
        "-d",
        "--database-dir",
        required=True,
        help="Directory containing the SQLite database file",
    )
    parser.add_argument("--port", type=int, default=8765, help="Preferred local HTTP port")
    return parser.parse_args()


def _parse_run(value: str) -> str:
    run = value.strip()
    if not run or not run.isdigit():
        raise argparse.ArgumentTypeError("run must contain only digits")
    return run


def _pick_port(preferred_port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", preferred_port))
        except OSError:
            sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _delayed_open(url: str) -> None:
    time.sleep(0.5)
    try:
        webbrowser.open(url)
    except Exception:
        print(f"Open {url} manually in a browser.", file=sys.stderr)
