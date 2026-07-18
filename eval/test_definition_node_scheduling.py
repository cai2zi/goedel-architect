"""Regression tests for Phase 2 definition-node scheduling."""
from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blueprint import Blueprint, BlueprintNode  # noqa: E402
from checkpoint import CheckpointState  # noqa: E402
from lean_compiler import _assemble_node_attempt  # noqa: E402
from orchestrator import NodeResult, OrchestratorResult, prove_dag  # noqa: E402
from pipeline import _assemble_final_file, _aux_lemma_decls, run_phase2  # noqa: E402
from prover import ProofSignal, ProverResult  # noqa: E402


def _definition(name: str, declaration: str) -> BlueprintNode:
    return BlueprintNode(
        name=name,
        kind="definition",
        statement="",
        proof_sketch="",
        lean_declaration=declaration,
    )


def _proof_node(
    name: str,
    declaration: str,
    dependencies: list[str],
    *,
    kind: str = "lemma",
) -> BlueprintNode:
    return BlueprintNode(
        name=name,
        kind=kind,
        statement="",
        proof_sketch="",
        dependencies=dependencies,
        lean_declaration=declaration,
    )


class DefinitionNodeSchedulingTest(unittest.TestCase):
    def setUp(self) -> None:
        base = _definition(
            "base_val",
            """@[blueprint (statement := /-- The base. -/)]
def base_val : ℝ := (30 : ℝ)
""",
        )
        height = _definition(
            "height_val",
            """@[blueprint (statement := /-- The height. -/)]
noncomputable def height_val : ℝ := (13 : ℝ) / 2
""",
        )
        mult = _proof_node(
            "mult_30_13_over_2",
            """@[blueprint
  (statement := /-- The product calculation. -/)
  (proof := /-- Normalize the definitions. -/)]
lemma mult_30_13_over_2 : base_val * height_val = 195 := by
  sorry_using [base_val, height_val]
""",
            ["base_val", "height_val"],
        )
        root = _proof_node(
            "mathd_algebra_478",
            """@[blueprint
  (statement := /-- The target. -/)
  (proof := /-- Use the calculation. -/)]
theorem mathd_algebra_478 : base_val * height_val = 195 := by
  sorry_using [mult_30_13_over_2]
""",
            ["mult_30_13_over_2"],
            kind="theorem",
        )
        self.blueprint = Blueprint(
            nodes=[base, height, mult, root],
            lean_file="\n".join(
                node.lean_declaration for node in [base, height, mult, root]
            ),
            target_theorem="mathd_algebra_478",
            fully_validated=True,
        )

    def test_definitions_skip_prover_and_rebuild_clean_context(self) -> None:
        calls: list[dict] = []

        def fake_prove_node(**kwargs) -> ProverResult:
            calls.append(kwargs)
            return ProverResult(signal=ProofSignal.SOLVED, proof_body="by norm_num")

        polluted_definition = (
            "@[blueprint] def base_val : ℝ := @[blueprint] "
            "def base_val : ℝ := (999 : ℝ)"
        )
        with patch("orchestrator.prove_node", side_effect=fake_prove_node):
            result = asyncio.run(
                prove_dag(
                    blueprint=self.blueprint,
                    compiler=object(),
                    retrieval=object(),
                    proved_cache={"base_val": polluted_definition},
                    node_timeout_s=None,
                )
            )

        self.assertEqual(
            [call["node_name"] for call in calls],
            ["mult_30_13_over_2", "mathd_algebra_478"],
        )
        self.assertTrue(result.all_proved())
        self.assertEqual(result.node_results["base_val"].result.proof_body, "")
        self.assertEqual(result.node_results["height_val"].result.proof_body, "")

        mult_call = calls[0]
        context = mult_call["parent_lemma_decls"]
        self.assertEqual(context.count("def base_val"), 1)
        self.assertEqual(context.count("def height_val"), 1)
        self.assertNotIn("@[blueprint", context)
        self.assertNotIn(":= @[blueprint", context)
        self.assertNotIn("999", context)
        self.assertEqual(mult_call["parent_proofs"], {})

        mult_node = self.blueprint.node_by_name("mult_30_13_over_2")
        assert mult_node is not None
        assembled = _assemble_node_attempt(
            mult_node.lean_declaration,
            context,
            "by norm_num",
        )
        self.assertEqual(assembled.count("def base_val"), 1)
        self.assertEqual(assembled.count("def height_val"), 1)
        self.assertNotIn(":= @[blueprint", assembled)
        self.assertIn(
            "lemma mult_30_13_over_2 : base_val * height_val = 195 := by norm_num",
            assembled,
        )

        root_call = calls[1]
        self.assertEqual(
            root_call["parent_proofs"],
            {"mult_30_13_over_2": "by norm_num"},
        )
        self.assertIn(
            "theorem mult_30_13_over_2 : base_val * height_val = 195 := by norm_num",
            root_call["parent_lemma_decls"],
        )

    def test_aux_and_final_output_keep_definition_rhs_once(self) -> None:
        proved_cache = {
            "base_val": "",
            "height_val": "",
            "mult_30_13_over_2": "by norm_num",
            "mathd_algebra_478": "by exact mult_30_13_over_2",
        }
        aux = _aux_lemma_decls(
            self.blueprint, proved_cache, "mathd_algebra_478",
        )

        self.assertEqual(aux.count("def base_val"), 1)
        self.assertEqual(aux.count("def height_val"), 1)
        self.assertIn("def base_val : ℝ := (30 : ℝ)", aux)
        self.assertIn("noncomputable def height_val : ℝ := (13 : ℝ) / 2", aux)
        self.assertNotIn("@[blueprint", aux)

        orch_result = OrchestratorResult(node_results={
            node.name: NodeResult(
                node=node,
                result=ProverResult(
                    signal=ProofSignal.SOLVED,
                    proof_body=proved_cache[node.name],
                ),
            )
            for node in self.blueprint.nodes
        })
        final_file = _assemble_final_file(self.blueprint, orch_result)
        self.assertEqual(final_file.count("def base_val"), 1)
        self.assertEqual(final_file.count("def height_val"), 1)
        self.assertIn("def base_val : ℝ := (30 : ℝ)", final_file)

    def test_phase2_rewrites_polluted_definition_checkpoint_entries(self) -> None:
        polluted = "@[blueprint] def base_val : ℝ := (999 : ℝ)"

        def fake_prove_node(**_kwargs) -> ProverResult:
            return ProverResult(signal=ProofSignal.SOLVED, proof_body="by norm_num")

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "checkpoint.json"
            state = CheckpointState(theorem_stmt="theorem mathd_algebra_478 : True")
            state.set_blueprint(self.blueprint)
            state.proved_cache = {
                "base_val": polluted,
                "height_val": "noncomputable def height_val : ℝ := 999",
            }
            state.proof_cache_keys = {
                "base_val": "old-signature-only-key",
                "height_val": "old-signature-only-key",
            }
            state.save(checkpoint_path)

            with patch("orchestrator.prove_node", side_effect=fake_prove_node):
                result = run_phase2(
                    checkpoint_path=checkpoint_path,
                    compiler=object(),
                    retrieval=object(),
                    node_timeout_s=None,
                )

            restored = CheckpointState.load(checkpoint_path)

        self.assertTrue(result.all_proved())
        self.assertEqual(restored.proved_cache["base_val"], "")
        self.assertEqual(restored.proved_cache["height_val"], "")
        for name in ("base_val", "height_val"):
            node = self.blueprint.node_by_name(name)
            assert node is not None
            self.assertEqual(restored.proof_cache_keys[name], node.cache_key())


if __name__ == "__main__":
    unittest.main()
