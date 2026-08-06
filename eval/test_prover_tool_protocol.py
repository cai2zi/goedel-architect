from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kimina_lean_compiler import CompilerResult  # noqa: E402
from prover import GoedelProver, ProofSignal, ProverResult  # noqa: E402


NODE_DECL = "theorem root : True := by sorry_using []"


class ToolCall:
    def __init__(self, call_id: str, name: str, arguments: dict | str) -> None:
        raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
        self.id = call_id
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=raw)

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class RecordingCompiler:
    def __init__(self, results: list[CompilerResult] | None = None, delay: float = 0) -> None:
        self.results = results
        self.delay = delay
        self.requests = []

    def check_many(self, requests):
        self.requests.append(list(requests))
        if self.delay:
            time.sleep(self.delay)
        if self.results is not None:
            return self.results[:len(requests)]
        return [CompilerResult(True) for _ in requests]


class Retrieval:
    def __init__(self, delay: float = 0) -> None:
        self.delay = delay
        self.calls = 0

    def search(self, query: str, k: int):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return []


def response(calls: list[ToolCall], content: str = ""):
    message = SimpleNamespace(tool_calls=calls, content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ProverToolProtocolTest(unittest.TestCase):
    def make_prover(self, retrieval=None, *, max_negation_probe_turns: int = 1) -> GoedelProver:
        with patch("prover.make_client", return_value=object()):
            return GoedelProver(
                "model",
                retrieval or Retrieval(),
                max_negation_probe_turns=max_negation_probe_turns,
                max_tool_calls_per_turn=3,
            )

    def prepare(self, prover, calls, limit=3, allowed=None):
        return prover._prepare_calls(
            calls,
            allowed or {"lean_compile", "step_lean_compile", "mathlib_search"},
            limit,
            "prove",
            NODE_DECL,
            "",
            "import Mathlib",
        )

    def test_four_hundred_calls_keep_only_first_three(self) -> None:
        prover = self.make_prover()
        calls = [ToolCall(str(i), "lean_compile", {"proof_body": f"by exact True.intro -- {i}"}) for i in range(400)]
        prepared, dropped = self.prepare(prover, calls)
        self.assertEqual([item.original_index for item in prepared], [0, 1, 2])
        self.assertEqual(len(dropped), 397)
        self.assertTrue(all(item["reason"] == "over_limit" for item in dropped))

        messages = []
        prover._process_response(
            response=response(calls),
            messages=messages,
            compiler=RecordingCompiler(),
            node_name="root",
            node_decl=NODE_DECL,
            parent_lemma_decls="",
            header="import Mathlib",
            turn=1,
            stage="prove",
            limit=3,
            allowed_names={"lean_compile"},
        )
        self.assertEqual(len(messages[0]["tool_calls"]), 3)
        self.assertEqual([message["role"] for message in messages], [
            "assistant", "tool", "tool", "tool",
        ])

    def test_same_turn_duplicates_are_removed(self) -> None:
        prover = self.make_prover()
        calls = [
            ToolCall("a", "lean_compile", {"proof_body": "by trivial"}),
            ToolCall("b", "lean_compile", {"proof_body": "by trivial"}),
        ]
        prepared, dropped = self.prepare(prover, calls)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(dropped[0]["reason"], "duplicate")

    def test_invalid_tool_arguments_are_dropped_before_history(self) -> None:
        prover = self.make_prover()
        prepared, dropped = self.prepare(prover, [
            ToolCall("missing", "lean_compile", {}),
            ToolCall("wrong", "step_lean_compile", {"lean_code": 12}),
            ToolCall("bad-k", "mathlib_search", {"query": "True", "k": 0}),
            ToolCall("valid", "lean_compile", {"proof_body": "by trivial"}),
        ])
        self.assertEqual([item.call.id for item in prepared], ["valid"])
        self.assertEqual([item["reason"] for item in dropped], [
            "invalid_arguments", "invalid_arguments", "invalid_arguments",
        ])
        self.assertTrue(all(len(item["hash"]) == 64 for item in dropped))

        _, malformed = self.prepare(prover, [ToolCall("json", "lean_compile", "{")])
        self.assertEqual(malformed[0]["reason"], "invalid_arguments")
        self.assertEqual(len(malformed[0]["hash"]), 64)

        messages = []
        compiler = RecordingCompiler()
        prover._process_response(
            response=response([ToolCall("missing", "lean_compile", {})]),
            messages=messages,
            compiler=compiler,
            node_name="root",
            node_decl=NODE_DECL,
            parent_lemma_decls="",
            header="import Mathlib",
            turn=1,
            stage="prove",
            limit=3,
            allowed_names={"lean_compile"},
        )
        self.assertEqual(messages, [])
        self.assertEqual(compiler.requests, [])

    def test_cross_turn_duplicate_returns_cache_without_compiling(self) -> None:
        prover = self.make_prover()
        compiler = RecordingCompiler()
        first, _ = self.prepare(prover, [ToolCall("a", "lean_compile", {"proof_body": "by trivial"})])
        first_outcome = prover._execute_calls(first, compiler, NODE_DECL, "", "import Mathlib")
        second, _ = self.prepare(prover, [ToolCall("b", "lean_compile", {"proof_body": "by trivial"})])
        second_outcome = prover._execute_calls(second, compiler, NODE_DECL, "", "import Mathlib")
        self.assertEqual(len(compiler.requests), 1)
        self.assertFalse(first_outcome[0].cache_hit)
        self.assertTrue(second_outcome[0].cache_hit)

    def test_compile_batch_and_search_execute_concurrently(self) -> None:
        retrieval = Retrieval(delay=0.15)
        prover = self.make_prover(retrieval)
        compiler = RecordingCompiler(delay=0.15)
        calls, _ = self.prepare(prover, [
            ToolCall("a", "lean_compile", {"proof_body": "by trivial"}),
            ToolCall("b", "step_lean_compile", {"lean_code": "import Mathlib\nexample : True := by trivial"}),
            ToolCall("c", "mathlib_search", {"query": "True"}),
        ])
        started = time.monotonic()
        outcomes = prover._execute_calls(calls, compiler, NODE_DECL, "", "import Mathlib")
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.27)
        self.assertEqual([item.call.call.id for item in outcomes], ["a", "b", "c"])
        self.assertEqual(len(compiler.requests[0]), 2)

    def test_step_success_does_not_solve_and_history_has_exact_pairs(self) -> None:
        prover = self.make_prover()
        compiler = RecordingCompiler([CompilerResult(True)])
        messages = []
        turn = prover._process_response(
            response=response([ToolCall("step", "step_lean_compile", {"lean_code": "import Mathlib\nexample : True := by trivial"})]),
            messages=messages,
            compiler=compiler,
            node_name="root",
            node_decl=NODE_DECL,
            parent_lemma_decls="",
            header="import Mathlib",
            turn=1,
            stage="prove",
            limit=3,
            allowed_names={"lean_compile", "step_lean_compile", "mathlib_search"},
        )
        self.assertEqual(turn.solved_proof, "")
        self.assertEqual([message["role"] for message in messages], ["assistant", "tool"])
        self.assertEqual(len(messages[0]["tool_calls"]), 1)

    def test_earliest_successful_canonical_call_wins(self) -> None:
        prover = self.make_prover()
        compiler = RecordingCompiler([CompilerResult(True), CompilerResult(True)])
        messages = []
        turn = prover._process_response(
            response=response([
                ToolCall("first", "lean_compile", {"proof_body": "by trivial"}),
                ToolCall("second", "lean_compile", {"proof_body": "by exact True.intro"}),
            ]),
            messages=messages,
            compiler=compiler,
            node_name="root",
            node_decl=NODE_DECL,
            parent_lemma_decls="",
            header="import Mathlib",
            turn=1,
            stage="prove",
            limit=3,
            allowed_names={"lean_compile"},
        )
        self.assertEqual(turn.solved_proof, "by trivial")

    def test_final_turn_filter_keeps_only_one_lean_compile(self) -> None:
        prover = self.make_prover()
        prepared, dropped = self.prepare(
            prover,
            [
                ToolCall("step", "step_lean_compile", {"lean_code": "import Mathlib"}),
                ToolCall("one", "lean_compile", {"proof_body": "by trivial"}),
                ToolCall("two", "lean_compile", {"proof_body": "by exact True.intro"}),
            ],
            limit=1,
            allowed={"lean_compile"},
        )
        self.assertEqual([item.call.id for item in prepared], ["one"])
        self.assertEqual([item["reason"] for item in dropped], ["not_allowed", "over_limit"])

    def test_diagnosis_contains_only_three_information_fields(self) -> None:
        block = ProverResult(
            ProofSignal.PROOF_TOO_HARD,
            "by omega",
            ['{"severity":"error","data":"No goals to be solved"}'],
        ).diagnosis_block()
        self.assertIn("## Signal", block)
        self.assertIn("## Proof body", block)
        self.assertIn("## Lean errors", block)
        self.assertNotIn("Analysis", block)
        self.assertNotIn("Suggested", block)

    def test_failed_compile_carries_goals_into_phase3_diagnosis(self) -> None:
        prover = self.make_prover()
        compiler = RecordingCompiler([
            CompilerResult(
                False,
                goals=["x : Nat\n⊢ x = x"],
                errors=["failed"],
                failure_kind="lean",
            ),
        ])
        turn = prover._process_response(
            response=response([
                ToolCall("compile", "lean_compile", {"proof_body": "by rfl"}),
            ]),
            messages=[],
            compiler=compiler,
            node_name="root",
            node_decl=NODE_DECL,
            parent_lemma_decls="",
            header="import Mathlib",
            turn=1,
            stage="prove",
            limit=3,
            allowed_names={"lean_compile"},
        )
        self.assertIn("failed", turn.last_errors)
        self.assertIn("x : Nat\n⊢ x = x", turn.last_errors)

    def test_negation_probe_can_be_disabled(self) -> None:
        prover = self.make_prover(max_negation_probe_turns=0)
        with patch.object(prover, "_chat") as chat:
            result = prover._probe_negation(
                RecordingCompiler(), "root", NODE_DECL, "", "import Mathlib",
            )
        self.assertIsNone(result)
        chat.assert_not_called()

    def test_negation_probe_retries_up_to_configured_limit(self) -> None:
        prover = self.make_prover(max_negation_probe_turns=3)
        failed = SimpleNamespace(solved_proof="")
        solved = SimpleNamespace(solved_proof="by trivial")
        with (
            patch.object(prover, "_chat", return_value=object()) as chat,
            patch.object(prover, "_process_turn", side_effect=[failed, solved]) as process,
        ):
            result = prover._probe_negation(
                RecordingCompiler(), "root", NODE_DECL, "", "import Mathlib",
            )
        self.assertEqual(result.signal, ProofSignal.FORMALLY_NEGATED)
        self.assertEqual(result.proof_body, "by trivial")
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(process.call_count, 2)
        self.assertEqual(process.call_args_list[0].kwargs["turn"], 1)
        self.assertEqual(process.call_args_list[1].kwargs["turn"], 2)


if __name__ == "__main__":
    unittest.main()
