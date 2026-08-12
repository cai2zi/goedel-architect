from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from blueprint_review_viewer.diff import whole_file_diff
from blueprint_review_viewer.review_schema import build_review_artifact, write_review_artifact
from blueprint_review_viewer.server import ReviewStore


class BlueprintReviewArtifactTest(unittest.TestCase):
    def _row(self, root: Path) -> dict:
        directory = root / "blueprints" / "case"
        directory.mkdir(parents=True)
        (directory / "generation_round_1.lean").write_text(
            '@[blueprint (title := "COT_STEP:S001")]\nlemma a : 1 = 1 := by sorry_using []\n'
            '@[blueprint (title := "COT_STEP:S002")]\ntheorem root : 1 = 1 := by sorry_using [a]\n', encoding="utf-8")
        (directory / "generation_round_2.lean").write_text(
            '@[blueprint (title := "COT_STEP:S001")]\nlemma a : 2 = 2 := by sorry_using []\n'
            '@[blueprint (title := "COT_STEP:S002")]\ntheorem root : 2 = 2 := by sorry_using [a]\n', encoding="utf-8")
        manifest = {"steps": [{"step_id": "S001", "source_start": 0, "source_end": 1, "source_text": "a", "source_sha256": "x"}, {"step_id": "S002", "source_start": 1, "source_end": 2, "source_text": "b", "source_sha256": "y"}]}
        trace = root / "traces" / "case.jsonl"; trace.parent.mkdir(parents=True)
        trace.write_text(json.dumps({"kind":"phase1GenerationEnd","round":1})+"\n", encoding="utf-8")
        return {"id":"case-1","source_id":"fixture/1","subset":"fixture","status":"semanticRejected", "blueprint_dir":str(directory), "trace_path":str(trace), "cot_manifest_json":json.dumps(manifest), "generation_validation":{"semanticAudit":{"classification":"semanticRejected"}}}

    def test_artifact_nodes_and_diff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._row(root)
            path, artifact = write_review_artifact(root, row)
            self.assertTrue(path.is_file())
            self.assertEqual(artifact["schemaVersion"], 1)
            self.assertEqual(len(artifact["candidates"]), 2)
            self.assertNotIn("edits", artifact)
            diff = whole_file_diff(artifact["candidates"][0], artifact["candidates"][1])
            self.assertIn("added", [row["right"]["kind"] for row in diff["rows"] if row["right"]])

    def test_store_builds_legacy_artifact_when_no_path_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); row = self._row(root)
            (root / "results.jsonl").write_text(json.dumps(row)+"\n", encoding="utf-8")
            store = ReviewStore(root)
            self.assertEqual(len(store.summaries()), 1)
            self.assertEqual(store.case("case-1")["source"]["source_id"], "fixture/1")


if __name__ == "__main__":
    unittest.main()
