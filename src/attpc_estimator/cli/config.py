from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_FILE = "config.toml"


def parse_toml_config(argv: list[str] | None) -> tuple[Path, dict[str, Any]]:
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
        raise SystemExit(
            f"config file must contain a TOML table: {config_path.resolve()}"
        )

    return config_path.resolve(), dict(payload)


def root_config_values(
    payload: dict[str, Any],
    *,
    allowed_keys: set[str],
) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key in allowed_keys}


def table_config_values(
    payload: dict[str, Any],
    *,
    table: str,
    allowed_keys: set[str],
) -> dict[str, Any]:
    raw_table: Any = payload
    for segment in table.split("."):
        if not isinstance(raw_table, dict):
            raise SystemExit(f"config section [{table}] must be a TOML table")
        if segment not in raw_table:
            return {}
        raw_table = raw_table.get(segment)
    if raw_table is None:
        return {}
    if not isinstance(raw_table, dict):
        raise SystemExit(f"config section [{table}] must be a TOML table")
    return {key: value for key, value in raw_table.items() if key in allowed_keys}


def parse_run(value: str) -> str:
    run = value.strip()
    if not run or not run.isdigit():
        raise argparse.ArgumentTypeError("run must contain only digits")
    return run


def argument_config_kwargs(values: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        "required": key not in values,
        "default": values.get(key),
    }


def bool_argument_config_kwargs(values: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        "action": argparse.BooleanOptionalAction,
        "required": key not in values,
        "default": values.get(key),
    }


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
