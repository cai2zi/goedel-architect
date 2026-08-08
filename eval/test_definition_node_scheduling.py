from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blueprint import _parse_blueprint  # noqa: E402
from checkpoint import CheckpointState, RunStatus  # noqa: E402
from kimina_lean_compiler import CompilerResult, KiminaInfrastructureError  # noqa: E402
from orchestrator import (  # noqa: E402
    NodeResult,
    OrchestratorResult,
    _prove_one,
    active_node_names,
    prove_dag,
)
from pipeline import (  # noqa: E402
    _assemble_final_file,
    _invalidate_stale_proofs,
    run_phase3,
    run_phase2_async,
)
from prover import ProofSignal, ProverResult  # noqa: E402
from refinement import _annotate_with_verdicts, _build_refinement_user_prompt  # noqa: E402


LEAN = """import Mathlib
import Architect

@[blueprint (statement := /-- A global definition. -/)]
def base : Nat := 1

@[blueprint (statement := /-- Used helper. -/) (proof := /-- Trivial. -/)]
lemma used : base = 1 := by sorry_using []

@[blueprint (statement := /-- Unrelated helper. -/) (proof := /-- Trivial. -/)]
lemma unused : True := by sorry_using []

@[blueprint (statement := /-- Root. -/) (proof := /-- Use helper. -/)]
theorem root : base = 1 := by sorry_using [used]
"""


POLICY_LEAN = """import Mathlib
import Architect

@[blueprint (statement := /-- First branch. -/) (proof := /-- Deferred. -/)]
lemma first : True := by sorry_using []

@[blueprint (statement := /-- Independent branch. -/) (proof := /-- Deferred. -/)]
lemma second : True := by sorry_using []

@[blueprint (statement := /-- Depends on the first branch. -/) (proof := /-- Deferred. -/)]
lemma after_first : True := by sorry_using [first]

@[blueprint (statement := /-- Root. -/) (proof := /-- Combine both branches. -/)]
theorem policy_root : True := by sorry_using [after_first, second]
"""


class FakeCompiler:
    def __init__(self, final_result: CompilerResult) -> None:
        self.final_result = final_result
        self.codes: list[str] = []

    def check(self, code: str, allow_sorry: bool = False) -> CompilerResult:
        self.codes.append(code)
        return self.final_result


def solved_result(blueprint) -> OrchestratorResult:
    results = {}
    for node in blueprint.nodes:
        if node.kind == "definition":
            proof = ""
        elif node.name == "used":
            proof = "by rfl"
        elif node.name == "root":
            proof = "by exact used"
        else:
            continue
        results[node.name] = NodeResult(node, ProverResult(ProofSignal.SOLVED, proof))
    return OrchestratorResult(
        results,
        active_node_names(blueprint),
        blueprint.target_theorem,
    )


class RootClosureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.blueprint = _parse_blueprint(LEAN, "root")

    def test_active_set_keeps_all_definitions_and_root_dependencies(self) -> None:
        self.assertEqual(active_node_names(self.blueprint), {"base", "used", "root"})

    def test_unrelated_proof_node_is_not_scheduled(self) -> None:
        attempted: list[str] = []

        async def fake_prove_one(**kwargs):
            name = kwargs["name"]
            attempted.append(name)
            node = self.blueprint.node_by_name(name)
            return NodeResult(node, ProverResult(ProofSignal.SOLVED, "by trivial"))

        async def run():
            with patch("orchestrator._prove_one", side_effect=fake_prove_one):
                return await prove_dag(
                    self.blueprint,
                    compiler=object(),
                    retrieval=object(),
                )

        result = asyncio.run(run())
        self.assertEqual(attempted, ["used", "root"])
        self.assertNotIn("unused", result.node_results)

    def test_every_definition_is_injected_into_node_context(self) -> None:
        with patch(
            "orchestrator.prove_node",
            return_value=ProverResult(ProofSignal.SOLVED, "by rfl"),
        ) as prove:
            with ThreadPoolExecutor(max_workers=1) as executor:
                asyncio.run(_prove_one(
                    name="used",
                    blueprint=self.blueprint,
                    proof_bodies={},
                    compiler=object(),
                    retrieval=object(),
                    model="model",
                    tracer=object(),
                    node_timeout_s=None,
                    llm_api_timeout_s=None,
                    node_max_prove_turns=1,
                    max_tool_calls_per_turn=3,
                    node_executor=executor,
                    node_semaphore=None,
                ))
        arguments = prove.call_args.kwargs
        self.assertEqual(len(arguments["definition_decls"]), 1)
        self.assertIn("def base", arguments["definition_decls"][0])
        self.assertIn("def base", arguments["parent_lemma_decls"])
        self.assertEqual(arguments["parent_signatures"], [])

    def test_final_file_drops_unrelated_lemma(self) -> None:
        assembled = _assemble_final_file(self.blueprint, solved_result(self.blueprint))
        self.assertIn("def base", assembled)
        self.assertIn("theorem used", assembled)
        self.assertIn("theorem root", assembled)
        self.assertNotIn("lemma unused", assembled)
        self.assertNotIn("sorry_using", assembled)
        self.assertNotIn("@[blueprint", assembled)

    def test_changed_root_invalidates_cached_proof(self) -> None:
        old_root = self.blueprint.node_by_name("root")
        cache = {"root": "by exact used", "unused": "by trivial"}
        keys = {"root": old_root.cache_key(), "unused": self.blueprint.node_by_name("unused").cache_key()}
        revised = _parse_blueprint(LEAN.replace("theorem root : base = 1", "theorem root : base = base"), "root")
        self.assertEqual(_invalidate_stale_proofs(revised, cache, keys), {})

    def test_phase3_writes_revised_root_declaration(self) -> None:
        revised = _parse_blueprint(
            LEAN.replace("theorem root : base = 1", "theorem root : base = base"),
            "root",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = CheckpointState("informal", "model", informal_proof="original COT")
            state.set_blueprint(self.blueprint)
            failed = solved_result(self.blueprint)
            failed.node_results["root"] = NodeResult(
                self.blueprint.node_by_name("root"),
                ProverResult(ProofSignal.PROOF_TOO_HARD, "by exact used", ["bad"]),
            )
            state.set_node_results(failed.node_results)
            state.save(path)
            with patch("pipeline.refine_blueprint", return_value=revised) as refine:
                run_phase3(
                    checkpoint_path=path,
                    compiler=FakeCompiler(CompilerResult(True)),
                )
            self.assertEqual(refine.call_args.kwargs["informal_statement"], "informal")
            self.assertEqual(refine.call_args.kwargs["informal_proof"], "original COT")
            loaded = CheckpointState.load(path)
        self.assertIn("theorem root : base = base", loaded.blueprint_lean_file)
        self.assertEqual(loaded.status, RunStatus.RUNNING)
        self.assertEqual(loaded.iteration, 1)
        self.assertNotIn("root", loaded.proved_cache)

    def test_checkpoint_round_trip_uses_new_schema(self) -> None:
        state = CheckpointState(
            "informal", "model", status=RunStatus.SOLVED, informal_proof="original COT",
        )
        state.set_blueprint(self.blueprint)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state.save(path)
            loaded = CheckpointState.load(path)
        self.assertTrue(loaded.root_proved)
        self.assertEqual(loaded.informal_statement, "informal")
        self.assertEqual(loaded.informal_proof, "original COT")

    def test_refinement_prompt_contains_original_cot(self) -> None:
        prompt = _build_refinement_user_prompt(
            "theorem root : True := by sorry",
            informal_statement="Original problem statement",
            informal_proof="Original chain of thought with final answer",
        )
        self.assertIn("## Original natural-language problem", prompt)
        self.assertIn("Original problem statement", prompt)
        self.assertIn("## Original COT / informal proof", prompt)
        self.assertIn("Original chain of thought with final answer", prompt)

    def test_final_verify_success_is_only_success_state(self) -> None:
        asyncio.run(self._run_final_case(CompilerResult(True), RunStatus.SOLVED, True))

    def test_final_verify_failure_removes_root_and_keeps_diagnosis(self) -> None:
        error = '{"severity":"error","pos":{"line":4,"column":2},"data":"bad"}'
        goal = '{"severity":"info","data":"x : Nat\\n⊢ x = x"}'
        state = asyncio.run(self._run_final_case(
            CompilerResult(False, errors=[error], goals=[goal], failure_kind="lean"),
            RunStatus.RUNNING,
            False,
        ))
        self.assertNotIn("root", state.proved_cache)
        self.assertEqual(state.node_results["root"]["proof_body"], "by exact used")
        self.assertEqual(state.node_results["root"]["lean_errors"], [error, goal])
        self.assertEqual(state.final_lean_errors, [error, goal])

    def test_phase3_retry_exhaustion_is_exhausted_not_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = CheckpointState("informal", "model")
            state.set_blueprint(self.blueprint)
            failed = solved_result(self.blueprint)
            failed.node_results["root"] = NodeResult(
                self.blueprint.node_by_name("root"),
                ProverResult(ProofSignal.PROOF_TOO_HARD, "by exact used", ["bad"]),
            )
            state.set_node_results(failed.node_results)
            state.save(path)
            with patch("pipeline.refine_blueprint", side_effect=RuntimeError("retry limit")):
                returned = run_phase3(
                    checkpoint_path=path,
                    compiler=FakeCompiler(CompilerResult(True)),
                )
            loaded = CheckpointState.load(path)
        self.assertEqual(returned.lean_file, self.blueprint.lean_file)
        self.assertEqual(loaded.status, RunStatus.EXHAUSTED)
        self.assertEqual(loaded.final_lean_errors, ["retry limit"])

    def test_phase3_kimina_failure_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = CheckpointState("informal", "model")
            state.set_blueprint(self.blueprint)
            state.set_node_results(solved_result(self.blueprint).node_results)
            state.save(path)
            with patch(
                "pipeline.refine_blueprint",
                side_effect=KiminaInfrastructureError("kimina unavailable"),
            ):
                run_phase3(
                    checkpoint_path=path,
                    compiler=FakeCompiler(CompilerResult(True)),
                )
            loaded = CheckpointState.load(path)
        self.assertEqual(loaded.status, RunStatus.ERROR)
        self.assertEqual(loaded.final_lean_errors, ["kimina unavailable"])

    def test_refinement_diagnosis_keeps_complete_error(self) -> None:
        failure = solved_result(self.blueprint)
        long_error = "E" * 1600 + "END"
        failure.node_results["root"] = NodeResult(
            self.blueprint.node_by_name("root"),
            ProverResult(ProofSignal.PROOF_TOO_HARD, "by exact used", [long_error]),
        )
        annotated = _annotate_with_verdicts(self.blueprint, failure)
        self.assertIn(long_error, annotated)
        self.assertNotIn("diagnosis truncated", annotated)

    async def _run_final_case(self, result, expected_status, expect_root) -> CheckpointState:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = CheckpointState("informal", "model")
            state.set_blueprint(self.blueprint)
            state.save(path)
            compiler = FakeCompiler(result)
            with patch("pipeline.prove_dag", new=AsyncMock(return_value=solved_result(self.blueprint))):
                await run_phase2_async(checkpoint_path=path, compiler=compiler, retrieval=object())
            loaded = CheckpointState.load(path)
        self.assertEqual(loaded.status, expected_status)
        self.assertEqual(loaded.root_proved, expect_root)
        self.assertEqual(len(compiler.codes), 1)
        return loaded


class ProofPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.blueprint = _parse_blueprint(POLICY_LEAN, "policy_root")

    def _run(self, policy: str) -> tuple[list[str], OrchestratorResult]:
        attempted: list[str] = []

        async def fake_prove_one(**kwargs):
            name = kwargs["name"]
            attempted.append(name)
            node = self.blueprint.node_by_name(name)
            if name == "first":
                return NodeResult(
                    node,
                    ProverResult(ProofSignal.PROOF_TOO_HARD, lean_errors=["first failed"]),
                )
            return NodeResult(node, ProverResult(ProofSignal.SOLVED, "by trivial"))

        async def run() -> OrchestratorResult:
            with patch("orchestrator._prove_one", side_effect=fake_prove_one):
                return await prove_dag(
                    self.blueprint,
                    compiler=object(),
                    retrieval=object(),
                    proof_policy=policy,
                )

        return attempted, asyncio.run(run())

    def test_first_failed_wave_finishes_wave_then_blocks_unrun_descendants(self) -> None:
        attempted, result = self._run("first_failed_wave")

        self.assertEqual(set(attempted), {"first", "second"})
        self.assertEqual(result.node_results["first"].result.signal, ProofSignal.PROOF_TOO_HARD)
        self.assertEqual(result.node_results["second"].result.signal, ProofSignal.SOLVED)
        for name in ("after_first", "policy_root"):
            self.assertEqual(
                result.node_results[name].result.signal,
                ProofSignal.BLOCKED_BY_DEPENDENCY,
            )
            self.assertIn(
                "Skipped by proof_policy=first_failed_wave",
                result.node_results[name].result.lean_errors[0],
            )

    def test_critical_path_stops_inside_wave_and_blocks_every_unrun_node(self) -> None:
        attempted, result = self._run("critical_path")

        self.assertEqual(attempted, ["first"])
        self.assertEqual(result.node_results["first"].result.signal, ProofSignal.PROOF_TOO_HARD)
        for name in ("second", "after_first", "policy_root"):
            self.assertEqual(
                result.node_results[name].result.signal,
                ProofSignal.BLOCKED_BY_DEPENDENCY,
            )
            self.assertIn(
                "Skipped by proof_policy=critical_path",
                result.node_results[name].result.lean_errors[0],
            )

    def test_full_policy_keeps_scheduling_independent_work(self) -> None:
        attempted, result = self._run("full")

        self.assertEqual(set(attempted), {"first", "second"})
        self.assertEqual(result.node_results["second"].result.signal, ProofSignal.SOLVED)
        self.assertEqual(
            result.node_results["after_first"].result.signal,
            ProofSignal.BLOCKED_BY_DEPENDENCY,
        )
        self.assertEqual(
            result.node_results["policy_root"].result.signal,
            ProofSignal.BLOCKED_BY_DEPENDENCY,
        )
        self.assertIn(
            "Unresolved dependencies",
            result.node_results["after_first"].result.lean_errors[0],
        )
        self.assertNotIn(
            "Skipped by proof_policy",
            result.node_results["after_first"].result.lean_errors[0],
        )


if __name__ == "__main__":
    unittest.main()
