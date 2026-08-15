from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from experiments.cot_blueprint_refine.export_semantic_review_bundle import (
    CHATGPT_PROMPT,
    build_rows,
    write_bundle,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class ExportSemanticReviewBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "demo_exp"
        self.blueprint_dir = self.root / "robustpa" / "blueprint" / "blueprints" / "one"
        self.blueprint_dir.mkdir(parents=True)
        self.blueprint = (
            "import Mathlib\nimport Architect\n"
            "@[blueprint] theorem n_final : 2 + 2 = 4 := by sorry_using []\n"
        )
        self.blueprint_path = self.blueprint_dir / "phase1_failed_last.lean"
        self.blueprint_path.write_text(self.blueprint, encoding="utf-8")
        self.blueprint_hash = hashlib.sha256(self.blueprint.encode()).hexdigest()
        _write_jsonl(
            self.root / "prepared" / "generation_inputs.jsonl",
            [{
                "name": "sample/1",
                "source": "demo",
                "row_index": 7,
                "problem": "What is 2+2?",
                "claimed_answer": "4",
                "post_think_cot": "Adding gives 4.",
            }],
        )
        _write_jsonl(
            self.root / "robustpa" / "blueprint" / "results.jsonl",
            [
                {
                    "source_id": "sample/ignored",
                    "status": "structuralRejected",
                },
                {
                    "source_id": "sample/1",
                    "record_id": "record-1",
                    "status": "semanticRejected",
                    "blueprint_dir": str(self.blueprint_dir),
                    "failed_blueprint_candidate_path": str(self.blueprint_path),
                    "semantic_audit_mode": "direct",
                    "semantic_comparator_protocol": "canonical_v2",
                    "trace_path": str(self.root / "trace.jsonl"),
                    "generation_history": [{
                        "round": 9,
                        "candidateHash": self.blueprint_hash,
                        "canonicalCandidateHash": self.blueprint_hash,
                        "semanticAuditInvoked": True,
                        "semanticAuditOrdinal": 8,
                        "validation": {"semanticAuditInvoked": True},
                    }],
                },
            ],
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_build_rows_selects_reject_and_hash_verified_final_blueprint(self) -> None:
        rows = build_rows(
            self.root,
            experiment_name="demo_exp",
            statuses={"semanticRejected"},
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["source_id"], "sample/1")
        self.assertEqual(row["problem"], "What is 2+2?")
        self.assertEqual(row["cot"], "Adding gives 4.")
        self.assertEqual(row["final_blueprint"], self.blueprint)
        self.assertEqual(row["final_blueprint_sha256"], self.blueprint_hash)
        self.assertEqual(row["final_generation_attempt"], 9)
        self.assertEqual(row["final_semantic_audit_ordinal"], 8)
        self.assertNotIn("semantic_errors", row)

    def test_hash_mismatch_is_rejected(self) -> None:
        results_path = self.root / "robustpa" / "blueprint" / "results.jsonl"
        rows = [json.loads(line) for line in results_path.read_text().splitlines()]
        rows[-1]["generation_history"][0]["canonicalCandidateHash"] = "0" * 64
        rows[-1]["generation_history"][0]["candidateHash"] = "0" * 64
        _write_jsonl(results_path, rows)
        with self.assertRaisesRegex(ValueError, "no candidate matches final semantic hash"):
            build_rows(
                self.root,
                experiment_name="demo_exp",
                statuses={"semanticRejected"},
            )

    def test_bundle_writes_readable_parquet_prompt_and_manifest(self) -> None:
        output = self.root / "bundle"
        manifest = write_bundle(
            self.root,
            experiment_name="demo_exp",
            output_dir=output,
            statuses={"semanticRejected"},
        )
        table = pq.read_table(output / "semantic_review_inputs.parquet")
        self.assertEqual(table.num_rows, 1)
        self.assertEqual(table.to_pylist()[0]["source_id"], "sample/1")
        self.assertEqual(manifest["rows"], 1)
        self.assertEqual(manifest["unique_source_ids"], 1)
        prompt = (output / "CHATGPT_PROMPT.md").read_text(encoding="utf-8")
        self.assertEqual(prompt.strip(), CHATGPT_PROMPT.strip())
        self.assertIn("infer the failure taxonomy", prompt)
        self.assertIn("Do not silently sample rows", prompt)
        self.assertNotIn("derivationShortcut", prompt)


if __name__ == "__main__":
    unittest.main()
