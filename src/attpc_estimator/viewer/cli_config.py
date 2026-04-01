from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_FILE = "config.toml"


def parse_toml_config(
    argv: list[str] | None,
    *,
    allowed_keys: set[str],
) -> tuple[Path, dict[str, Any]]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    config_file = _parse_config_option(raw_argv)
    config_path = Path(config_file).expanduser()

    if not config_path.exists():
        if config_file != DEFAULT_CONFIG_FILE:
            raise SystemExit(f"config file not found: {config_path.resolve()}")
        return config_path.resolve(), {}

    if not config_path.is_file():
        raise SystemExit(f"config path is not a file: {config_path.resolve()}")

    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)

    if not isinstance(payload, dict):
        raise SystemExit(f"config file must contain a TOML table: {config_path.resolve()}")

    config = {
        key: value
        for key, value in dict(payload).items()
        if key in allowed_keys
    }

    return config_path.resolve(), config


def _parse_config_option(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        default=DEFAULT_CONFIG_FILE,
    )
    namespace, _ = parser.parse_known_args(argv)
    return str(namespace.config_file)
