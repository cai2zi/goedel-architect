# Canonical location is now src/tracer.py — this file kept for backward compat.
"""
Lightweight event tracer for the Goedel-Architect proving loop.

Events are emitted at each tool call / result inside _run_loop so that
graph_viz.py can build an interactive visualization of the proving steps.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    kind: str          # theorem_start | tool_call | tool_result | model_text | final_verify
    thm_name: str
    turn: int = 0
    call_id: str | None = None
    tool_name: str | None = None
    args: dict[str, Any] | None = None
    result: str | None = None
    ok: bool | None = None
    ts: float = field(default_factory=time.time)


class NullTracer:
    def emit(self, event: TraceEvent) -> None:
        pass


class JsonlTracer:
    """Thread-safe JSONL writer — one line per event."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._closed = False
        self._f = self.path.open("a", encoding="utf-8", buffering=1)

    def emit(self, event: TraceEvent) -> None:
        line = json.dumps(asdict(event)) + "\n"
        with self._lock:
            if self._closed:
                return
            self._f.write(line)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._f.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
