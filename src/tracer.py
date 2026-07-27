"""
Lightweight event tracer for the Goedel-Architect proving loop.

Events are emitted at each tool call / result inside _run_loop so that
graph_viz.py can build an interactive visualization of the proving steps.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    kind: str          # theorem_start | tool_call | tool_result | model_text | final_verify
    thm_name: str
    turn: int = 0
    call_id: str | None = None
    tool_name: str | None = None
    phase: str | None = None
    iteration: int | None = None
    args: dict[str, Any] | None = None
    result: str | None = None
    ok: bool | None = None
    ts: float = field(default_factory=time.time)


class NullTracer:
    def emit(self, event: TraceEvent) -> None:
        pass

    def with_context(self, *, phase: str | None = None, iteration: int | None = None) -> ContextTracer:
        return ContextTracer(self, phase=phase, iteration=iteration)


class ContextTracer:
    """Attach phase/iteration metadata to events emitted by nested code."""

    def __init__(self, base, *, phase: str | None = None, iteration: int | None = None) -> None:
        self.base = base
        self.phase = phase
        self.iteration = iteration

    def emit(self, event: TraceEvent) -> None:
        if event.phase is None or event.iteration is None:
            event = replace(
                event,
                phase=event.phase if event.phase is not None else self.phase,
                iteration=event.iteration if event.iteration is not None else self.iteration,
            )
        self.base.emit(event)

    def with_context(self, *, phase: str | None = None, iteration: int | None = None) -> ContextTracer:
        return ContextTracer(
            self.base,
            phase=self.phase if phase is None else phase,
            iteration=self.iteration if iteration is None else iteration,
        )


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

    def with_context(self, *, phase: str | None = None, iteration: int | None = None) -> ContextTracer:
        return ContextTracer(self, phase=phase, iteration=iteration)

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
