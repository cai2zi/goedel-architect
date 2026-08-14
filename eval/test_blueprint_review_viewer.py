from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from blueprint_review_viewer.diff import whole_file_diff
from blueprint_review_viewer.review_schema import build_review_artifact, write_review_artifact
from blueprint_review_viewer.server import ExperimentCatalog, ReviewStore, experiment_root


class BlueprintReviewArtifactTest(unittest.TestCase):
    def _row(self, root: Path) -> dict:
        directory = root / "blueprints" / "case"
        directory.mkdir(parents=True)
        round_1 = (
            '@[blueprint (title := "COT_STEP:S001")]\nlemma a : 1 = 1 := by sorry_using []\n'
            '@[blueprint (title := "COT_STEP:S002")]\ntheorem root : 1 = 1 := by sorry_using [a]\n'
        )
        round_2 = (
            '@[blueprint (title := "COT_STEP:S001")]\nlemma a : 2 = 2 := by sorry_using []\n'
            '@[blueprint (title := "COT_STEP:S002")]\ntheorem root : 2 = 2 := by sorry_using [a]\n'
        )
        (directory / "generation_round_1.lean").write_text(round_1, encoding="utf-8")
        (directory / "generation_round_2.lean").write_text(round_2, encoding="utf-8")
        manifest = {"steps": [{"step_id": "S001", "source_start": 0, "source_end": 1, "source_text": "a", "source_sha256": "x"}, {"step_id": "S002", "source_start": 1, "source_end": 2, "source_text": "b", "source_sha256": "y"}]}
        trace = root / "traces" / "case.jsonl"; trace.parent.mkdir(parents=True)
        trace.write_text(
            "\n".join([
                json.dumps({
                    "kind": "llm_response", "turn": 1,
                    "args": {
                        "phase": "phase1", "reasoning_content": "builder think 1",
                        "tool_calls": [], "finish_reason": "tool_calls",
                    },
                }),
                json.dumps({"kind": "phase1GenerationEnd", "round": 1}),
                json.dumps({
                    "kind": "llm_response", "turn": 2,
                    "args": {
                        "phase": "phase1", "reasoning_content": "builder think 2",
                        "tool_calls": [], "finish_reason": "tool_calls",
                    },
                }),
            ]) + "\n",
            encoding="utf-8",
        )
        return {
            "id": "case-1", "source_id": "fixture/1", "subset": "fixture",
            "status": "semanticRejected", "blueprint_dir": str(directory),
            "trace_path": str(trace), "cot_manifest_json": json.dumps(manifest),
            "generation_validation": {"semanticAudit": {"classification": "semanticRejected"}},
            "generation_history": [
                {
                    "round": 1, "candidateHash": hashlib.sha256(round_1.encode()).hexdigest(),
                    "inputTokens": 100, "maxCompletionTokens": 200,
                    "deterministicErrors": [{
                        "stage": "canonical_lean", "code": "canonicalLean",
                        "nodeName": "", "message": "unknown declaration",
                        "diagnosticFingerprint": "det-1",
                    }],
                    "semanticErrors": [], "warnings": [],
                    "validation": {
                        "mechanicalStageReached": "canonical_lean",
                        "mechanicalFailureStage": "canonical_lean",
                        "wholeFileLeanSuccess": False,
                        "canonicalLeanSuccess": False,
                        "leanErrors": ["unknown declaration"],
                        "canonicalLeanErrors": ["unknown declaration"],
                        "phase2StructuralErrors": [],
                        "phase2StandaloneErrors": [],
                        "phase2StandaloneSummary": {
                            "checkedNodeCount": 0, "cachedNodeCount": 0,
                            "failedNodeCount": 0, "notRunReason": "canonical_lean",
                            "durationMs": 0,
                        },
                        "semanticAuditInvoked": False,
                    },
                },
                {
                    "round": 2, "candidateHash": hashlib.sha256(round_2.encode()).hexdigest(),
                    "inputTokens": 120, "maxCompletionTokens": 180,
                    "deterministicErrors": [],
                    "semanticErrors": [{
                        "stage": "whole_cot_comparator", "code": "dependencyFidelity",
                        "nodeName": "a", "message": "connect the source use-chain",
                        "diagnosticFingerprint": "sem-1",
                    }],
                    "warnings": [],
                    "validation": {
                        "mechanicalStageReached": "static_shadow",
                        "mechanicalFailureStage": None,
                        "wholeFileLeanSuccess": True,
                        "canonicalLeanSuccess": True,
                        "leanErrors": [], "canonicalLeanErrors": [],
                        "phase2StructuralErrors": [], "phase2StandaloneErrors": [],
                        "phase2StandaloneSummary": {
                            "checkedNodeCount": 2, "cachedNodeCount": 0,
                            "failedNodeCount": 0, "notRunReason": "", "durationMs": 4.5,
                        },
                        "semanticAuditInvoked": True,
                        "semanticAuditMode": "separate",
                        "semanticActualRequestCount": 2,
                        "semanticAudit": {
                            "mode": "separate", "actualRequestCount": 2,
                            "classification": "semanticRejected",
                            "formalDecompiler": {
                                "reasoning_content": "decompile think",
                                "raw_content": "decompile answer",
                            },
                            "wholeCotComparator": {
                                "reasoning_content": "compact think",
                                "raw_content": "compact answer",
                            },
                        },
                    },
                },
            ],
        }

    def test_artifact_nodes_and_diff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._row(root)
            path, artifact = write_review_artifact(root, row)
            self.assertTrue(path.is_file())
            self.assertEqual(artifact["schemaVersion"], 3)
            self.assertEqual(len(artifact["candidates"]), 2)
            self.assertNotIn("edits", artifact)
            self.assertEqual(len(artifact["generationRounds"]), 2)
            first, second = artifact["generationRounds"]
            self.assertTrue(first["candidateHashMatches"])
            self.assertEqual(first["feedback"]["deterministic"]["status"], "failed")
            self.assertEqual(first["feedback"]["semantic"]["status"], "notRun")
            self.assertEqual(
                first["feedback"]["deterministic"]["phase2Standalone"]["status"],
                "notRun",
            )
            self.assertEqual(second["feedback"]["deterministic"]["status"], "passed")
            self.assertEqual(second["feedback"]["semantic"]["status"], "failed")
            self.assertEqual(
                second["feedback"]["semantic"]["errors"][0]["code"],
                "dependencyFidelity",
            )
            self.assertEqual(
                second["artifacts"]["decompileAnswer"],
                {
                    "available": True,
                    "thinking": "decompile think",
                    "answer": "decompile answer",
                },
            )
            self.assertEqual(
                second["artifacts"]["compactAnswer"]["thinking"],
                "compact think",
            )
            self.assertEqual(
                second["artifacts"]["builderInput"]["previousBlueprint"],
                artifact["candidates"][0]["lean"],
            )
            self.assertEqual(
                second["artifacts"]["builderInput"]["deterministicErrors"][0]["code"],
                "canonicalLean",
            )
            self.assertEqual(
                second["artifacts"]["builderAnswer"]["thinking"],
                "builder think 2",
            )
            self.assertEqual(
                second["artifacts"]["builderAnswer"]["answer"],
                artifact["candidates"][1]["lean"],
            )
            diff = whole_file_diff(artifact["candidates"][0], artifact["candidates"][1])
            self.assertIn("added", [row["right"]["kind"] for row in diff["rows"] if row["right"]])

    def test_store_builds_artifact_when_no_path_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); row = self._row(root)
            (root / "results.jsonl").write_text(json.dumps(row)+"\n", encoding="utf-8")
            store = ReviewStore(root)
            self.assertEqual(len(store.summaries()), 1)
            self.assertEqual(store.case("case-1")["source"]["source_id"], "fixture/1")

    def test_store_rebuilds_stale_schema_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._row(root)
            stale = root / "blueprints" / "case" / "review.json"
            stale.write_text(json.dumps({"schemaVersion": 1, "source": {"id": "wrong"}}), encoding="utf-8")
            row["review_artifact_path"] = str(stale)
            (root / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            artifact = ReviewStore(root).case("case-1")
            self.assertEqual(artifact["schemaVersion"], 3)
            self.assertEqual(len(artifact["generationRounds"]), 2)

    def test_store_retries_an_in_flight_trailing_jsonl_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._row(root)
            second = dict(first)
            second.update({"id": "case-2", "source_id": "fixture/2"})
            results = root / "results.jsonl"
            complete_first = json.dumps(first) + "\n"
            complete_second = json.dumps(second) + "\n"
            split = len(complete_second) // 2
            results.write_text(complete_first + complete_second[:split], encoding="utf-8")

            store = ReviewStore(root)
            self.assertEqual([item["id"] for item in store.summaries()], ["case-1"])
            self.assertEqual(store.diagnostics()[0]["kind"], "pendingTail")

            with results.open("a", encoding="utf-8") as handle:
                handle.write(complete_second[split:])
            self.assertEqual(
                [item["id"] for item in store.summaries()],
                ["case-1", "case-2"],
            )
            self.assertEqual(store.diagnostics(), [])

    def test_store_skips_but_reports_a_complete_invalid_jsonl_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._row(root)
            (root / "results.jsonl").write_text(
                json.dumps(row) + "\n" + '{"id":"broken"\n',
                encoding="utf-8",
            )
            store = ReviewStore(root)
            self.assertEqual(len(store.summaries()), 1)
            warning = store.diagnostics()[0]
            self.assertEqual(warning["kind"], "invalidResultRow")
            self.assertEqual(warning["line"], 2)

    def test_semantic_execution_error_and_missing_candidate_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._row(root)
            row["generation_history"][1]["semanticErrors"] = []
            row["generation_history"][1]["validation"]["semanticAuditError"] = {
                "stage": "formal_decompiler_or_comparator",
                "code": "semanticAuditError", "message": "response truncated",
            }
            row["generation_history"].append({
                "round": 3, "candidateHash": "",
                "deterministicErrors": [{
                    "stage": "parse_basic", "code": "phase1ToolCallCount",
                    "nodeName": "", "message": "expected one tool call",
                }],
                "semanticErrors": [], "warnings": [],
                "validation": {},
            })
            artifact = build_review_artifact(root, row)
            second = artifact["generationRounds"][1]
            third = artifact["generationRounds"][2]
            self.assertEqual(second["feedback"]["semantic"]["status"], "executionError")
            self.assertFalse(third["candidateAvailable"])
            self.assertIsNone(third["candidateHashMatches"])
            self.assertEqual(
                third["feedback"]["deterministic"]["phase2Standalone"]["status"],
                "notRun",
            )
            self.assertEqual(third["feedback"]["semantic"]["status"], "notRun")

    def test_phase1b_history_is_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._row(root)
            row["phase1b_edit_history"] = [{"round": 1, "accepted": [{"nodeName": "a"}]}]
            artifact = build_review_artifact(root, row)
            self.assertNotIn("edits", artifact)
            self.assertNotIn("phase1b", json.dumps(artifact).lower())

    def test_standalone_issue_keeps_derived_error_details(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._row(root)
            first = row["generation_history"][0]
            first["deterministicErrors"] = [{
                "stage": "phase2_standalone", "code": "phase2StandaloneFailed",
                "nodeName": "root", "message": "Unknown identifier `missing`",
            }]
            first["validation"].update({
                "mechanicalStageReached": "phase2_standalone",
                "mechanicalFailureStage": "phase2_standalone",
                "wholeFileLeanSuccess": True,
                "canonicalLeanSuccess": True,
                "leanErrors": [], "canonicalLeanErrors": [],
                "phase2StandaloneErrors": [{
                    "code": "phase2StandaloneFailed", "nodeName": "root",
                    "errorKind": "unknownIdentifier", "identifiers": ["missing"],
                    "diagnostic": "Unknown identifier `missing`",
                    "preflightHash": "standalone-hash", "originDeclaration": "root",
                }],
                "phase2StandaloneSummary": {
                    "checkedNodeCount": 2, "cachedNodeCount": 0,
                    "failedNodeCount": 1, "notRunReason": "", "durationMs": 3.0,
                },
            })
            feedback = build_review_artifact(root, row)["generationRounds"][0]["feedback"]
            self.assertEqual(feedback["deterministic"]["wholeGraph"]["status"], "passed")
            standalone = feedback["deterministic"]["phase2Standalone"]
            self.assertEqual(standalone["status"], "failed")
            self.assertEqual(standalone["issues"][0]["errorKind"], "unknownIdentifier")
            self.assertEqual(standalone["issues"][0]["originDeclaration"], "root")

    def test_experiment_name_resolves_to_blueprint_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            self.assertEqual(
                experiment_root("my_experiment", base),
                (base / "my_experiment" / "robustpa" / "blueprint").resolve(),
            )

    def test_experiment_name_rejects_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for name in ("../other", "nested/experiment", ".", ""):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    experiment_root(name, base)

    def test_catalog_lists_only_experiments_with_blueprint_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            valid = base / "valid_exp" / "robustpa" / "blueprint"
            valid.mkdir(parents=True)
            (valid / "results.jsonl").write_text("", encoding="utf-8")
            (base / "missing_results" / "robustpa" / "blueprint").mkdir(parents=True)
            (base / "plain_directory").mkdir()
            catalog = ExperimentCatalog(base)
            self.assertEqual(catalog.experiment_names(), ["valid_exp"])
            self.assertEqual(catalog.store("valid_exp").root, valid.resolve())
            with self.assertRaises(ValueError):
                catalog.store("missing_results")


if __name__ == "__main__":
    unittest.main()
