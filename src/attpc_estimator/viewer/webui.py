from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

import uvicorn

from .app_factory import create_app
from .cli_config import parse_toml_config
from .services import ViewerService


def main() -> None:
    args = _parse_args()
    trace_path = Path(args.trace_path).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()

    if not trace_path.exists():
        raise SystemExit(f"trace path not found: {trace_path}")
    if not workspace.is_dir():
        raise SystemExit(f"workspace not found: {workspace}")

    service = ViewerService(trace_path=trace_path, workspace=workspace)
    frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    app = create_app(service, frontend_dist)

    port = _pick_port(args.port)
    url = f"http://0.0.0.0:{port}"
    print(f"Trace Viewer running at {url}")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def _parse_args() -> argparse.Namespace:
    config_path, config = parse_toml_config(
        sys.argv[1:],
        allowed_keys={
            "trace_path",
            "workspace",
            "port",
        },
    )
    parser = argparse.ArgumentParser(description="Launch the trace viewer app")
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        default=str(config_path),
        help="Path to a TOML config file. Defaults to config.toml.",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        required="workspace" not in config,
        default=config.get("workspace"),
        help="Workspace directory containing histogram artifacts, filters, and the labels database",
    )
    parser.add_argument(
        "-t",
        "--trace-path",
        required="trace_path" not in config,
        default=config.get("trace_path"),
        help="Path to a trace file or a directory containing run_<run>.h5 files",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.get("port", 8766),
        help="Preferred local HTTP port",
    )
    return parser.parse_args()


def _pick_port(preferred_port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", preferred_port))
        except OSError:
            sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    main()
