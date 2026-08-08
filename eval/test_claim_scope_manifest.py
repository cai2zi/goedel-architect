from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from cot_blueprint_refine.claim_scope_manifest import (  # noqa: E402
    ClaimScopeValidationError,
    make_claim_scope_manifest,
    unassigned_spans,
)
from cot_blueprint_refine.llm_claim_scope import (  # noqa: E402
    build_claim_scope_messages,
    claim_scope_quality_issues,
    parse_claim_scope_annotation,
)
from cot_blueprint_refine.llm_cot_splitter import atomize_cot  # noqa: E402
from blueprint import _parse_blueprint, _render_step_grounded_proof  # noqa: E402
from semantic_fidelity import parse_cot_manifest, validate_blueprint_fidelity  # noqa: E402


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class ClaimScopeManifestTest(unittest.TestCase):
    def fixture(self) -> dict:
        source = "If x is positive: x > 0. Thus x + 1 > 1. Finally x + 2 > 2."
        scope_text = "If x is positive: "
        first = "x > 0. "
        second = "Thus x + 1 > 1. "
        third = "Finally x + 2 > 2."
        claims = []
        cursor = len(scope_text)
        for index, text in enumerate((first, second, third), start=1):
            claims.append({
                "claim_id": f"C{index:03d}", "source_start": cursor,
                "source_end": cursor + len(text), "source_text": text,
                "source_sha256": sha(text), "scope_ids": ["G001"],
            })
            cursor += len(text)
        scopes = [{
            "scope_id": "G001", "scope_type": "case_condition",
            "applies_to_claim_ids": ["C001", "C002", "C003"],
            "source_start": 0, "source_end": len(scope_text),
            "source_text": scope_text, "source_sha256": sha(scope_text),
        }]
        return make_claim_scope_manifest(source, claims=claims, scopes=scopes)

    def test_minimal_manifest_has_no_step_or_atom_layer(self) -> None:
        manifest = self.fixture()
        self.assertEqual(set(manifest), {
            "schema", "schema_version", "source_text", "source_sha256", "claims", "scopes",
        })
        self.assertFalse(any("atom" in key or "step" in key for key in manifest))
        self.assertEqual(unassigned_spans(manifest), [])

    def test_rejects_one_use_scope_and_bad_reverse_link(self) -> None:
        manifest = self.fixture()
        one = copy.deepcopy(manifest)
        one["scopes"][0]["applies_to_claim_ids"] = ["C001"]
        with self.assertRaisesRegex(ClaimScopeValidationError, "at least two"):
            make_claim_scope_manifest(one["source_text"], claims=one["claims"], scopes=one["scopes"])
        broken = copy.deepcopy(manifest)
        broken["claims"][1]["scope_ids"] = []
        with self.assertRaisesRegex(ClaimScopeValidationError, "reverse scope links"):
            make_claim_scope_manifest(broken["source_text"], claims=broken["claims"], scopes=broken["scopes"])

    def test_llm_annotation_builds_exact_global_claim_scope_spans(self) -> None:
        source = "Under x > 0, both facts hold. First x + 1 > 1. Second x + 2 > 2."
        atoms = atomize_cot(source)
        self.assertGreaterEqual(len(atoms), 3)
        raw = (
            "[[CLAIM_SCOPE_V1]]\n"
            + json.dumps([
                {"kind": "scope", "scope_type": "shared_assumption", "through": atoms[-1]["atom_id"], "end": atoms[0]["atom_id"]},
                {"kind": "claim", "end": atoms[1]["atom_id"]},
                {"kind": "claim", "end": atoms[-1]["atom_id"]},
            ])
            + "\n[[/CLAIM_SCOPE_V1]]"
        )
        manifest, segments, warnings = parse_claim_scope_annotation(raw, source, atoms)
        self.assertEqual([claim["claim_id"] for claim in manifest["claims"]], ["C001", "C002"])
        self.assertEqual(manifest["scopes"][0]["applies_to_claim_ids"], ["C001", "C002"])
        self.assertEqual(warnings, [])
        self.assertEqual(segments[-1]["source_end"], len(source))
        prompt = build_claim_scope_messages(source, atoms)[0]["content"]
        self.assertIn("may be wrong", prompt)
        self.assertIn("at least two", prompt.lower())

    def test_single_use_scope_is_losslessly_folded_into_claim(self) -> None:
        source = "In the first case, x is positive. We get x = 1. Therefore y = 2."
        atoms = atomize_cot(source)
        raw = "[[CLAIM_SCOPE_V1]]\n" + json.dumps([
            {"kind": "scope", "scope_type": "case_condition", "through": atoms[1]["atom_id"], "end": atoms[0]["atom_id"]},
            {"kind": "claim", "end": atoms[1]["atom_id"]},
            {"kind": "claim", "end": atoms[-1]["atom_id"]},
        ]) + "\n[[/CLAIM_SCOPE_V1]]"
        manifest, _segments, warnings = parse_claim_scope_annotation(raw, source, atoms)
        self.assertEqual(manifest["scopes"], [])
        self.assertEqual(manifest["claims"][0]["source_start"], 0)
        self.assertIn("merged_single_use_scope_into_C001", warnings)

    def test_blueprint_render_and_binding_use_claim_ids(self) -> None:
        manifest = self.fixture()
        rendered = _render_step_grounded_proof(json.dumps(manifest), include_ir=True)
        self.assertIn("[COT_SCOPE G001", rendered)
        self.assertIn("applies_to=C001..C003", rendered)
        self.assertIn("[COT_CLAIM C003", rendered)
        contract = parse_cot_manifest(manifest)
        self.assertEqual(contract.final_step_id, "C003")
        lean = """import Mathlib
@[blueprint (title := "COT_CLAIM:C001") (statement := /-- a -/)]
lemma a : (0:Nat) = 0 := by sorry_using []
@[blueprint (title := "COT_CLAIM:C002") (statement := /-- b -/) (proof := /-- a -/)]
lemma b : (1:Nat) = 1 := by sorry_using [a]
@[blueprint (title := "COT_CLAIM:C003") (statement := /-- c -/) (proof := /-- b -/)]
theorem target : (2:Nat) = 2 := by sorry_using [b]
"""
        blueprint = _parse_blueprint(lean, "target")
        issues = validate_blueprint_fidelity(
            blueprint, contract, claimed_answer="2", require_step_bindings=True,
        )
        self.assertFalse([issue for issue in issues if issue.category == "binding"], issues)

    def test_quality_gate_rejects_layout_but_keeps_false_assertions(self) -> None:
        source = "### Step 1\nWe compute:\nx = 6."
        parts = ("### Step 1", "We compute:", "x = 6.")
        claims = []
        cursor = 0
        for index, part in enumerate(parts, start=1):
            start = source.index(part, cursor)
            end = start + len(part)
            claims.append({
                "claim_id": f"C{index:03d}", "source_start": start,
                "source_end": end, "source_text": part,
                "source_sha256": sha(part), "scope_ids": [],
            })
            cursor = end
        checked = make_claim_scope_manifest(source, claims=claims, scopes=[])
        issues = claim_scope_quality_issues(checked)
        self.assertEqual(
            [issue["code"] for issue in issues],
            ["HEADING_ONLY_CLAIM", "DANGLING_LEAD_IN_CLAIM"],
        )
        self.assertNotIn("C003", {issue["claim_id"] for issue in issues})

    def test_parser_merges_dangling_lead_in_into_following_claim(self) -> None:
        source = "We compute:\n\nx = 6."
        atoms = atomize_cot(source)
        self.assertEqual(len(atoms), 2)
        raw = "[[CLAIM_SCOPE_V1]]\n" + json.dumps([
            {"kind": "claim", "end": atoms[0]["atom_id"]},
            {"kind": "claim", "end": atoms[1]["atom_id"]},
        ]) + "\n[[/CLAIM_SCOPE_V1]]"
        manifest, _segments, warnings = parse_claim_scope_annotation(raw, source, atoms)
        self.assertEqual(len(manifest["claims"]), 1)
        self.assertEqual(manifest["claims"][0]["source_text"], source)
        self.assertIn("merged_dangling_lead_in_into_next_claim", warnings)


if __name__ == "__main__":
    unittest.main()
