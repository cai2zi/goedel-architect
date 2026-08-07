from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tracer import JsonlTracer, TraceEvent  # noqa: E402


class JsonlTracerTest(unittest.TestCase):
    def test_v2_event_has_monotonic_identity_and_duration_fields(self) -> None:
        event = TraceEvent(kind="tool_call", thm_name="root", span_id="span")
        self.assertEqual(event.schema_version, 2)
        self.assertTrue(event.event_id)
        self.assertEqual(event.span_id, "span")
        self.assertGreater(event.wall_time_ns, 0)
        self.assertLessEqual(event.monotonic_ns, time.monotonic_ns())

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
