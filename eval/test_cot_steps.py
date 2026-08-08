from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from checkpoint import CheckpointState  # noqa: E402
from cot_blueprint_refine.cot_steps import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    SUBCLAIM_BUILDER_VERSION,
    build_cot_steps_from_sections,
    decode_steps,
    encode_steps,
    render_numbered_cot,
    split_cot_steps,
)
from cot_blueprint_refine.cot_manifest_validation import (  # noqa: E402
    OFFSET_SPACE,
    validate_split_manifest,
)
from cot_blueprint_refine.llm_cot_splitter import (  # noqa: E402
    atomize_cot,
    sections_from_boundaries,
)
from cot_blueprint_refine.prepare_inputs import (  # noqa: E402
    DATASET_SUBSET,
    make_generation_row,
    prepare,
)
from robustpa_refine.run_robustpa_refine import _make_record  # noqa: E402


class CotStepSplitterTest(unittest.TestCase):
    def test_narration_only_opener_is_context_but_problem_givens_are_substantive(self) -> None:
        source = (
            "To solve the problem, we analyze the geometric configuration involving a torus.\n\n"
            "### Step 1: Setup\n\n"
            "We are given a torus of major radius 6 and minor radius 3.\n\n"
            "### Final Answer\n\n\\boxed{7}"
        )
        steps = split_cot_steps(source)

        self.assertEqual(steps[0]["role"], "context")
        self.assertFalse(steps[0]["requires_formalization"])
        self.assertTrue(steps[1]["requires_formalization"])
        self.assertEqual(steps[1]["depends_on"], [])
        self.assertEqual(steps[1]["claims"][0]["claim_id"], "S002.C001")
        self.assertIn("requires_formalization=false", render_numbered_cot(steps))

        substantive = split_cot_steps(
            "To solve this geometry problem, we are given an equilateral triangle "
            "of side length 6 and must find x."
        )
        self.assertTrue(substantive[0]["requires_formalization"])

        numbered_givens = split_cot_steps(
            "To solve the given geometric problem, we analyze the configuration. "
            "We are given DG = 3 and are asked to find the length of AE."
        )
        self.assertTrue(numbered_givens[0]["requires_formalization"])

        relational_setup = split_cot_steps(
            "To solve the problem, we analyze circles tangent to the sides of triangle ABC."
        )
        self.assertTrue(relational_setup[0]["requires_formalization"])

    def test_task_narration_is_context_inside_a_substantive_step(self) -> None:
        source = (
            "### Step 1: Compute\n\n"
            "We have x = 3.\n\n"
            "So the key is to determine y.\n\n"
            "Therefore y = 4."
        )
        step = split_cot_steps(source)[0]
        self.assertEqual(
            [claim["claim_id"] for claim in step["claims"]],
            ["S001.C001", "S001.C002"],
        )
        self.assertNotIn("key is to determine", " ".join(
            claim["source_text"] for claim in step["claims"]
        ))
    def test_heading_ordinals_are_not_semantic_numbers(self) -> None:
        source = r"""### Step 12: Substitute x = 7

We obtain x + 2 = 9.

---

### Step 13: Final Answer

Therefore \boxed{9}."""

        steps = split_cot_steps(source)

        self.assertEqual([step["step_id"] for step in steps], ["S001", "S002"])
        self.assertNotIn("12", steps[0]["numbers"])
        self.assertNotIn("13", steps[1]["numbers"])
        self.assertIn("7", steps[0]["numbers"])
        self.assertIn("2", steps[0]["numbers"])
        self.assertIn("9", steps[1]["numbers"])
        self.assertEqual(steps[1]["role"], "conclusion")

    def test_bare_markdown_ordinal_is_not_a_semantic_number(self) -> None:
        steps = split_cot_steps("### 3. Compute 4 + 5\n\nThe value is 9.")

        self.assertEqual(steps[0]["numbers"], ["4", "5", "9"])

    def test_split_and_encoding_are_deterministic(self) -> None:
        source = "# Solution\n\n### Setup\nLet x = 1.\n\n### Final Answer\n\\boxed{1}"

        first = split_cot_steps(source)
        second = split_cot_steps(source)

        self.assertEqual(first, second)
        self.assertEqual(decode_steps(encode_steps(first)), first)
        rendered = render_numbered_cot(first)
        self.assertIn("[COT_STEP S001", rendered)
        self.assertIn("[COT_STEP S002", rendered)
        self.assertIn("Let x = 1.", rendered)

    def test_unstructured_text_uses_paragraph_fallback(self) -> None:
        source = "Let x be a real number.\n\nCompute x + 1 = 3.\n\nThus x = 2."
        steps = split_cot_steps(source)

        self.assertGreaterEqual(len(steps), 1)
        self.assertEqual(steps[-1]["role"], "conclusion")
        self.assertEqual(
            [step["depends_on"] for step in steps],
            [[]] + [[f"S{index:03d}"] for index in range(1, len(steps))],
        )


class StructuredSubclaimBuilderTest(unittest.TestCase):
    def test_narration_opener_with_attached_separator_is_context(self) -> None:
        source = (
            "To solve this geometry problem, we analyze the diagram and the given "
            "conditions step by step:\n\n---\n\n"
        )
        sections = sections_from_boundaries(
            source,
            atomize_cot(source),
            [atomize_cot(source)[-1]["atom_id"]],
        )
        steps = build_cot_steps_from_sections(
            sections,
            structured_subclaims=True,
            splitter_mode="unit",
        )

        self.assertFalse(steps[0]["requires_formalization"])
        self.assertEqual(steps[0]["claims"], [])
        self.assertEqual(steps[0]["segments"][0]["kind"], "context")

    def build_one_step(self, source: str) -> list[dict]:
        atoms = atomize_cot(source)
        sections = sections_from_boundaries(source, atoms, [atoms[-1]["atom_id"]])
        steps = build_cot_steps_from_sections(
            sections,
            structured_subclaims=True,
            splitter_mode="llm-cot-boundary-v4",
        )
        validate_split_manifest(source, steps)
        return steps

    def test_layout_list_table_and_intro_math_form_exact_subclaims(self) -> None:
        source = """### Work
The equation is:
$$
x = 2
$$
- First, x = 1.
- Next, y = 2.
| name | value |
|---|---|
| x | 1 |
| y | 2 |
Ordinary explanation. It continues here."""
        step = self.build_one_step(source)[0]

        self.assertEqual(step["manifest_schema_version"], MANIFEST_SCHEMA_VERSION)
        self.assertEqual(step["subclaim_builder_version"], SUBCLAIM_BUILDER_VERSION)
        self.assertEqual(step["offset_space"], OFFSET_SPACE)
        self.assertEqual(
            "".join(
                step["source_text"][segment["source_start"]:segment["source_end"]]
                for segment in step["segments"]
            ),
            source,
        )
        self.assertEqual(
            [claim["claim_kind"] for claim in step["claims"]],
            [
                "display_math",
                "list_item",
                "list_item",
                "table_data_row",
                "table_data_row",
                "prose",
                "prose",
            ],
        )
        contexts = [
            segment for segment in step["segments"] if segment["kind"] == "context"
        ]
        self.assertEqual(
            [segment["context_type"] for segment in contexts],
            ["heading", "narration", "table_layout"],
        )
        self.assertEqual(contexts[1]["scope_type"], "claim_prefix")
        self.assertEqual(len(step["atoms"]), len(atomize_cot(source)))
        self.assertEqual(
            [atom["atom_id"] for atom in step["atoms"]],
            step["atom_ids"],
        )

    def test_numbered_bold_title_is_context_even_with_ordinal_atom_split(self) -> None:
        source = "2. **Second Triangle**:\nWe obtain x = 2."
        step = self.build_one_step(source)[0]

        first = step["segments"][0]
        self.assertEqual(first["kind"], "context")
        self.assertEqual(first["context_type"], "heading")
        self.assertEqual(
            step["source_text"][first["source_start"]:first["source_end"]].strip(),
            "2. **Second Triangle**:",
        )
        self.assertNotIn("Second Triangle", " ".join(
            claim["source_text"] for claim in step["claims"]
        ))

    def test_short_labels_and_intros_are_context_but_following_math_is_claim(self) -> None:
        source = r"""### Work
- **Area**:
$$
A = 6
$$
Here's why:
The triangles overlap.
Key observations about $G_{n,k}$:
- If k is even, parity is preserved.
Let's compute the total:
$$
S = 10
$$"""
        step = self.build_one_step(source)[0]

        context_texts = [
            step["source_text"][segment["source_start"]:segment["source_end"]].strip()
            for segment in step["segments"]
            if segment["kind"] == "context"
        ]
        self.assertIn("- **Area**:", context_texts)
        self.assertIn("Here's why:", context_texts)
        self.assertIn(r"Key observations about $G_{n,k}$:", context_texts)
        self.assertIn("Let's compute the total:", context_texts)

        claim_text = "\n".join(claim["source_text"] for claim in step["claims"])
        self.assertIn("A = 6", claim_text)
        self.assertIn("The triangles overlap.", claim_text)
        self.assertIn("If k is even, parity is preserved.", claim_text)
        self.assertIn("S = 10", claim_text)

    def test_intro_shaped_assertions_remain_claims(self) -> None:
        source = r"""### Work
- **Area = 6**:
Here's why x = 2:
Key observations about x > 0:
Let's compute 2 + 3 = 5:
Let's compute whether x is even:
**The triangles are congruent**:"""
        step = self.build_one_step(source)[0]

        claim_text = "\n".join(claim["source_text"] for claim in step["claims"])
        for expected in (
            "- **Area = 6**:",
            "Here's why x = 2:",
            "Key observations about x > 0:",
            "Let's compute 2 + 3 = 5:",
            "Let's compute whether x is even:",
            "**The triangles are congruent**:",
        ):
            self.assertIn(expected, claim_text)

    def test_case_condition_is_shared_scope_for_all_following_claims(self) -> None:
        source = "### Case 1: k = 4\nWe obtain x = 2. Therefore y = 3."
        step = self.build_one_step(source)[0]

        self.assertEqual(len(step["claims"]), 2)
        scope = next(segment for segment in step["segments"] if segment.get("scope_id"))
        self.assertEqual(scope["scope_type"], "case_condition")
        self.assertIn("k = 4", source[scope["source_start"]:scope["source_end"]])
        self.assertEqual(
            scope["applies_to_claim_ids"],
            ["S001.C001", "S001.C002"],
        )
        self.assertEqual(step["scope_count"], 1)
        self.assertEqual(step["claims"][0]["scope_ids"], ["S001.G001"])
        self.assertEqual(step["claims"][1]["scope_ids"], ["S001.G001"])

    def test_colon_prefix_is_scope_not_standalone_claim(self) -> None:
        source = "We are told:\n- n = 1.\n- n = 2."
        step = self.build_one_step(source)[0]

        self.assertEqual(
            [claim["source_text"].strip() for claim in step["claims"]],
            ["- n = 1.", "- n = 2."],
        )
        scope = next(segment for segment in step["segments"] if segment.get("scope_id"))
        self.assertEqual(scope["scope_type"], "claim_prefix")
        self.assertEqual(
            scope["applies_to_claim_ids"],
            ["S001.C001", "S001.C002"],
        )

    def test_adjacent_case_scopes_do_not_cross_into_next_case(self) -> None:
        source = (
            "### Case 1: k = 1\nWe obtain x = 1.\n"
            "### Case 2: k = 2\nWe obtain x = 2."
        )
        step = self.build_one_step(source)[0]
        scopes = [segment for segment in step["segments"] if segment.get("scope_id")]

        self.assertEqual(len(scopes), 2)
        self.assertEqual(scopes[0]["applies_to_claim_ids"], ["S001.C001"])
        self.assertEqual(scopes[1]["applies_to_claim_ids"], ["S001.C002"])
        self.assertEqual(step["claims"][0]["scope_ids"], ["S001.G001"])
        self.assertEqual(step["claims"][1]["scope_ids"], ["S001.G002"])

    def test_direct_prefix_scope_stops_at_the_next_prefix(self) -> None:
        source = "We are given:\nx = 1.\nNow compute:\ny = 2."
        step = self.build_one_step(source)[0]
        scopes = [segment for segment in step["segments"] if segment.get("scope_id")]

        self.assertEqual(len(scopes), 2)
        self.assertEqual(scopes[0]["applies_to_claim_ids"], ["S001.C001"])
        self.assertEqual(scopes[1]["applies_to_claim_ids"], ["S001.C002"])

    def test_outer_prefix_scope_covers_nested_cases(self) -> None:
        source = (
            "Consider the following cases:\n"
            "### Case 1: k = 1\nWe obtain x = 1.\n"
            "### Case 2: k = 2\nWe obtain x = 2."
        )
        step = self.build_one_step(source)[0]
        scopes = [segment for segment in step["segments"] if segment.get("scope_id")]

        self.assertEqual(scopes[0]["scope_type"], "claim_prefix")
        self.assertEqual(
            scopes[0]["applies_to_claim_ids"], ["S001.C001", "S001.C002"]
        )
        self.assertEqual(scopes[1]["applies_to_claim_ids"], ["S001.C001"])
        self.assertEqual(scopes[2]["applies_to_claim_ids"], ["S001.C002"])

    def test_consecutive_prose_atoms_remain_independent_claims(self) -> None:
        source = "We are given x = 1. Therefore y = 2. Hence z = 3."
        step = self.build_one_step(source)[0]

        self.assertEqual(len(step["claims"]), 3)
        self.assertEqual(
            [claim["atom_ids"] for claim in step["claims"]],
            [["A0001"], ["A0002"], ["A0003"]],
        )

    def test_duplicate_identical_list_claims_keep_distinct_ids_and_spans(self) -> None:
        source = "- x = 1.\n- x = 1.\n- y = 2."
        step = self.build_one_step(source)[0]
        first, second = step["claims"][:2]

        self.assertEqual(first["source_text"], second["source_text"])
        self.assertEqual(first["source_sha256"], second["source_sha256"])
        self.assertNotEqual(first["claim_id"], second["claim_id"])
        self.assertNotEqual(first["source_start"], second["source_start"])

    def test_decode_with_source_runs_canonical_validation(self) -> None:
        source = "### Work\nx = 1."
        steps = self.build_one_step(source)
        decoded = decode_steps(encode_steps(steps), source=source)
        self.assertEqual(decoded, steps)

        decoded[0]["segments"][0]["source_end"] -= 1
        with self.assertRaisesRegex(ValueError, "segment"):
            decode_steps(encode_steps(decoded), source=source)


class CotManifestPreparationTest(unittest.TestCase):
    def test_generation_row_keeps_original_proof_and_adds_manifest(self) -> None:
        post_think = "### Step 1: Compute\n2 + 2 = 4.\n\n### Final Answer\n\\boxed{4}"
        row = {"ID": "id", "source": "demo", "row_index": 1, "problem": "2+2?"}

        generated = make_generation_row(row, post_think, "4")

        self.assertEqual(generated["informal_proof"], post_think)
        self.assertEqual(generated["post_think_cot"], post_think)
        steps = decode_steps(generated["cot_manifest_json"])
        self.assertEqual(len(steps), 2)
        self.assertNotIn("1", steps[0]["numbers"])

    def test_prepare_persists_manifest_in_jsonl_and_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "predictions.jsonl"
            raw_cot = (
                "<think>private</think>\n"
                "### Step 1: Compute\n2 + 2 = 4.\n\n"
                "### Final Answer\n\\boxed{4}"
            )
            input_path.write_text(json.dumps({
                "ID": "demo/1",
                "source": "demo",
                "row_index": 1,
                "problem": "What is 2+2?",
                "status": "ok",
                "finish_reason": "stop",
                "raw_cot": raw_cot,
            }) + "\n", encoding="utf-8")
            config = OmegaConf.create({
                "input_predictions": str(input_path),
                "output_base": str(root / "outputs"),
                "exp_name": "unit",
                "include_ids": [],
            })

            prepare(config)

            prepared = root / "outputs" / "unit" / "prepared"
            jsonl_row = json.loads(
                (prepared / "generation_inputs.jsonl").read_text(encoding="utf-8").strip()
            )
            parquet_path = next((prepared / "data" / DATASET_SUBSET).glob("*.parquet"))
            parquet_row = pq.read_table(parquet_path).to_pylist()[0]
            self.assertEqual(jsonl_row["informal_proof"], parquet_row["informal_proof"])
            self.assertEqual(jsonl_row["cot_manifest_json"], parquet_row["cot_manifest_json"])
            self.assertEqual(len(decode_steps(parquet_row["cot_manifest_json"])), 2)


class CotManifestPersistenceTest(unittest.TestCase):
    def test_old_checkpoint_and_record_get_empty_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path = root / "old.json"
            checkpoint_path.write_text(json.dumps({
                "informal_statement": "statement",
                "model": "model",
            }), encoding="utf-8")

            checkpoint = CheckpointState.load(checkpoint_path)
            record = _make_record(
                "subset", "split", root / "data.parquet", 1,
                {"name": "row", "informal_statement": "statement"},
            )

        self.assertEqual(checkpoint.cot_manifest_json, "")
        self.assertEqual(checkpoint.claimed_answer, "")
        self.assertEqual(record.cot_manifest_json, "")
        self.assertEqual(record.claimed_answer, "")

    def test_checkpoint_and_record_preserve_manifest_and_claim(self) -> None:
        manifest = encode_steps(split_cot_steps("### Final Answer\n\\boxed{7}"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path = root / "state.json"
            CheckpointState(
                "statement",
                "model",
                informal_proof="proof",
                cot_manifest_json=manifest,
                claimed_answer="7",
            ).save(checkpoint_path)
            checkpoint = CheckpointState.load(checkpoint_path)
            record = _make_record(
                "subset", "split", root / "data.parquet", 1,
                {
                    "name": "row",
                    "informal_statement": "statement",
                    "informal_proof": "proof",
                    "cot_manifest_json": manifest,
                    "claimed_answer": "7",
                },
            )

        self.assertEqual(checkpoint.cot_manifest_json, manifest)
        self.assertEqual(checkpoint.claimed_answer, "7")
        self.assertEqual(record.cot_manifest_json, manifest)
        self.assertEqual(record.claimed_answer, "7")


if __name__ == "__main__":
    unittest.main()
