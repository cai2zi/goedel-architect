from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    # Serialize before opening the destination.  A malformed in-memory row
    # must not create or touch a misleading empty/partial results file.
    payload = json.dumps(row, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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


def unlink_if_exists(path: Path) -> None:
    path.unlink(missing_ok=True)
