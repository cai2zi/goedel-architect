from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tracer import JsonlTracer, TraceEvent  # noqa: E402


class JsonlTracerTest(unittest.TestCase):
    def test_emit_after_close_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            tracer = JsonlTracer(trace_path)

            tracer.emit(TraceEvent(kind="before_close", thm_name="root", ok=True))
            tracer.close()
            tracer.emit(TraceEvent(kind="after_close", thm_name="root", ok=False))
            tracer.close()

            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["kind"] for row in rows], ["before_close"])


if __name__ == "__main__":
    unittest.main()
