from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable


def load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when --config points to a YAML file.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return data


def apply_config_environment(config: dict[str, Any]) -> None:
    env = config.get("environment") or {}
    if not isinstance(env, dict):
        raise ValueError("Config key 'environment' must be an object.")
    for key, value in env.items():
        if value is not None:
            os.environ.setdefault(str(key), str(value))


def config_path_from_argv(argv: Iterable[str], default: Path) -> Path | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=default)
    args, _unknown = parser.parse_known_args(list(argv))
    return args.config


def add_config_arg(parser: argparse.ArgumentParser, default: Path) -> None:
    parser.add_argument("--config", type=Path, default=default)


def set_defaults_from_config(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    *,
    aliases: dict[str, str] | None = None,
    ignore: set[str] | None = None,
) -> None:
    aliases = aliases or {}
    ignore = ignore or set()
    for action in parser._actions:
        if action.dest == argparse.SUPPRESS or action.dest in ignore:
            continue
        key = aliases.get(action.dest, action.dest)
        if key in config:
            action.default = _coerce_default(config[key], action.type)


def _coerce_default(value: Any, action_type: Any) -> Any:
    if action_type is Path and value is not None:
        return Path(value)
    return value
