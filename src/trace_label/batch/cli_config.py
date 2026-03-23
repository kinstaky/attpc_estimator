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
    section_names: tuple[str, ...],
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

    config = _extract_section(payload, section_names=section_names)
    unknown_keys = sorted(set(config) - allowed_keys)
    if unknown_keys:
        key_list = ", ".join(unknown_keys)
        raise SystemExit(f"unsupported config option(s) in {config_path.resolve()}: {key_list}")

    return config_path.resolve(), config


def _parse_config_option(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-c",
        "--connfig",
        "--config",
        dest="config_file",
        default=DEFAULT_CONFIG_FILE,
    )
    namespace, _ = parser.parse_known_args(argv)
    return str(namespace.config_file)


def _extract_section(payload: dict[str, Any], *, section_names: tuple[str, ...]) -> dict[str, Any]:
    for section_name in section_names:
        section = payload.get(section_name)
        if section is not None:
            if not isinstance(section, dict):
                raise SystemExit(f"config section [{section_name}] must be a TOML table")
            return dict(section)

    return {key: value for key, value in payload.items() if not isinstance(value, dict)}
