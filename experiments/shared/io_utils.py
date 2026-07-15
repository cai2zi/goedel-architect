from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rows_by_id(path: Path, id_key: str = "id") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        row_id = row.get(id_key)
        if row_id is not None:
            out[str(row_id)] = row
    return out


def unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def safe_stem(text: str, prefix: str = "") -> str:
    stem = re.sub(r"[^A-Za-z0-9_]", "_", text)
    stem = re.sub(r"_+", "_", stem).strip("_")
    if prefix and not stem.startswith(prefix):
        stem = f"{prefix}{stem}"
    if not stem:
        stem = f"{prefix}item"
    if not re.match(r"[A-Za-z_]", stem):
        stem = f"{prefix}{stem}"
    return stem


def model_dir_name(model: str) -> str:
    return safe_stem(model.replace("-", "_").replace(".", "_"))


def default_output_root(repo_root: Path, experiment: str, model: str) -> Path:
    return repo_root.parent / "czx_work" / "goedel-architect" / experiment / model_dir_name(model)


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)

