from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

from cot_blueprint_refine.shared_phase1a import validate  # noqa: E402


class SharedPhase1ASeedBarrierTest(unittest.TestCase):
    def _fixture(self, statuses: dict[str, str]) -> Path:
        temporary = Path(tempfile.mkdtemp())
        prepared = temporary / "prepared" / "generation_inputs.jsonl"
        result_root = temporary / "robustpa" / "blueprint"
        prepared.parent.mkdir(parents=True)
        result_root.mkdir(parents=True)
        prepared.write_text("".join(
            json.dumps({"name": source_id}) + "\n" for source_id in statuses
        ))
        rows = []
        for index, (source_id, status) in enumerate(statuses.items()):
            row = {
                "source_id": source_id,
                "status": status,
                "subset": "subset",
                "split": "split",
                "record_id": f"record-{index}",
                "theorem_name": f"root_{index}",
                "claimed_answer": str(index),
                "cot_manifest_json": json.dumps({"steps": [source_id]}),
                "failed_blueprint_failure_stage": "fixtureFailure",
            }
            rows.append(row)
            if status == "phase1aReady":
                path = (
                    result_root / "blueprints" / "subset" / "split"
                    / f"record-{index}" / "phase1a_canonical.lean"
                )
                path.parent.mkdir(parents=True)
                path.write_text(f"theorem root_{index} : True := by trivial\n")
        (result_root / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        return temporary

    def test_fixed_failures_are_dropped_and_v2_manifest_is_published(self) -> None:
        root = self._fixture({
            "ready/1": "phase1aReady",
            "drop/1": "structuralRejected",
            "ready/2": "phase1aReady",
        })
        target = validate(root, 3, drop_source_ids={"drop/1"})
        payload = json.loads(target.read_text())
        include = json.loads((target.parent / "phase1a_ready_ids.json").read_text())
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual((payload["totalCount"], payload["readyCount"], payload["droppedCount"]), (3, 2, 1))
        self.assertEqual(include, {"include_ids": ["ready/1", "ready/2"]})
        self.assertEqual(payload["dropped"][0]["source_id"], "drop/1")

    def test_unexpected_fourth_failure_still_blocks_publication(self) -> None:
        root = self._fixture({
            "ready/1": "phase1aReady",
            "drop/1": "structuralRejected",
            "unexpected/1": "structuralRejected",
        })
        with self.assertRaisesRegex(RuntimeError, "unexpected/1"):
            validate(root, 3, drop_source_ids={"drop/1"})


if __name__ == "__main__":
    unittest.main()
