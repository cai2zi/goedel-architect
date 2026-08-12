from __future__ import annotations

import json
import sys
import time
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kimina_lean_compiler import CompilerResult  # noqa: E402
from prover import LEAN_COMPILE_TOOL, GoedelProver, ProofSignal, ProverResult  # noqa: E402


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


class RecordingTracer:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


def response(calls: list[ToolCall], content: str = "", finish_reason: str | None = None):
    message = SimpleNamespace(tool_calls=calls, content=content)
    return SimpleNamespace(choices=[SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
    )])


class ProverToolProtocolTest(unittest.TestCase):
    def make_prover(
        self, retrieval=None, *, tracer=None, max_negation_probe_turns: int = 1,
    ) -> GoedelProver:
        with patch("prover.make_client", return_value=object()):
            return GoedelProver(
                "model",
                retrieval or Retrieval(),
                tracer=tracer,
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

    def test_length_finish_upgrades_only_this_prover_to_eight_k(self) -> None:
        tracer = RecordingTracer()
        with (
            patch.dict(
                "os.environ",
                {
                    "GOEDEL_PROVER_MAX_TOKENS": "4096",
                    "GOEDEL_PROVER_LENGTH_RETRY_MAX_TOKENS": "8192",
                },
            ),
            patch("prover.make_client", return_value=object()),
        ):
            prover = GoedelProver("model", Retrieval(), tracer=tracer)
        truncated = response([], finish_reason="length")
        completed = response(
            [ToolCall("proof", "lean_compile", {"proof_body": "by trivial"})],
            finish_reason="tool_calls",
        )
        with patch(
            "prover.chat_completion_with_retry",
            side_effect=[truncated, completed, completed],
        ) as chat:
            actual = prover._chat([], "root", 1, "prove", [LEAN_COMPILE_TOOL], "prove_node")
            prover._chat([], "root", 2, "prove", [LEAN_COMPILE_TOOL], "prove_node")
        self.assertIs(actual, completed)
        self.assertEqual(
            [call.kwargs["max_completion_tokens"] for call in chat.call_args_list],
            [4096, 8192, 8192],
        )
        self.assertEqual(
            len([event for event in tracer.events if event.kind == "llm_length_retry"]),
            1,
        )

    def test_phase2_uses_explicit_qwen_thinking_sampling(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "GOEDEL_PROVER_ENABLE_THINKING": "true",
                    "GOEDEL_PROVER_TEMPERATURE": "0.6",
                    "GOEDEL_PROVER_TOP_P": "0.95",
                    "GOEDEL_PROVER_TOP_K": "20",
                    "GOEDEL_PROVER_MIN_P": "0.0",
                    "GOEDEL_PROVER_PRESENCE_PENALTY": "0.0",
                    "GOEDEL_PROVER_REPETITION_PENALTY": "1.0",
                },
            ),
            patch("prover.make_client", return_value=object()),
        ):
            prover = GoedelProver("model", Retrieval())
        completed = response(
            [ToolCall("proof", "lean_compile", {"proof_body": "by trivial"})],
            finish_reason="tool_calls",
        )
        with patch("prover.chat_completion_with_retry", return_value=completed) as chat:
            prover._chat([], "root", 1, "prove", [LEAN_COMPILE_TOOL], "prove_node")
        kwargs = chat.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.6)
        self.assertEqual(kwargs["top_p"], 0.95)
        self.assertEqual(kwargs["presence_penalty"], 0.0)
        self.assertEqual(kwargs["extra_body"], {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True},
        })
        self.assertTrue(kwargs["trace_args"]["enable_thinking"])

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

    def test_parallel_tool_spans_use_each_tools_actual_duration(self) -> None:
        tracer = RecordingTracer()
        prover = self.make_prover(Retrieval(delay=0.1), tracer=tracer)
        compiler = RecordingCompiler()
        cached, _ = self.prepare(prover, [
            ToolCall("warm", "lean_compile", {"proof_body": "by trivial"}),
        ])
        prover._execute_calls(cached, compiler, NODE_DECL, "", "import Mathlib")
        tracer.events.clear()

        prover._process_response(
            response=response([
                ToolCall("cached", "lean_compile", {"proof_body": "by trivial"}),
                ToolCall("search", "mathlib_search", {"query": "True"}),
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
            allowed_names={"lean_compile", "mathlib_search"},
        )
        results = {
            event.call_id: event for event in tracer.events if event.kind == "tool_result"
        }
        self.assertLess(results["cached"].duration_ms, 20)
        self.assertGreaterEqual(results["search"].duration_ms, 80)
        self.assertEqual(
            {event.span_id for event in tracer.events if event.kind == "tool_call"},
            {event.span_id for event in tracer.events if event.kind == "tool_result"},
        )

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

    def test_repeated_tool_diagnostics_are_compacted_only_for_model_history(self) -> None:
        tracer = RecordingTracer()
        prover = self.make_prover(tracer=tracer)
        repeated = '{"severity":"error","data":"No goals to be solved"}'
        middle = "M" * 4000
        tail = '{"severity":"error","data":"Unknown constant at tail"}'
        compiler = RecordingCompiler([
            CompilerResult(
                False,
                errors=[repeated] * 729 + [middle, tail],
                failure_kind="lean",
            ),
        ])
        messages = []
        with patch.dict("os.environ", {"GOEDEL_TOOL_FEEDBACK_MAX_CHARS": "512"}):
            turn = prover._process_response(
                response=response([
                    ToolCall("compile", "lean_compile", {"proof_body": "by omega"}),
                ]),
                messages=messages,
                compiler=compiler,
                node_name="root",
                node_decl=NODE_DECL,
                parent_lemma_decls="",
                header="import Mathlib",
                turn=2,
                stage="prove",
                limit=1,
                allowed_names={"lean_compile"},
            )

        model_feedback = messages[-1]["content"]
        self.assertLessEqual(len(model_feedback), 512)
        self.assertEqual(model_feedback.count(repeated), 1)
        self.assertIn("previous line repeated 728 additional times", model_feedback)
        self.assertIn("tool feedback truncated", model_feedback)
        self.assertIn("Unknown constant at tail", model_feedback)

        full_trace = next(event for event in tracer.events if event.kind == "tool_result")
        self.assertEqual(full_trace.result.count(repeated), 729)
        self.assertGreater(len(full_trace.result), len(model_feedback))
        compacted = next(
            event for event in tracer.events if event.kind == "tool_feedback_compacted"
        )
        self.assertEqual(compacted.args["duplicate_lines_removed"], 728)
        self.assertTrue(compacted.args["truncated"])
        self.assertEqual(turn.last_errors.count(repeated), 1)
        self.assertIn(
            "[previous diagnostic repeated 728 additional times]",
            turn.last_errors,
        )

    def test_prover_history_rolls_to_latest_legal_tool_pair(self) -> None:
        class SequentialCompiler:
            def __init__(self):
                self.results = [
                    CompilerResult(False, errors=["first failure"], failure_kind="lean"),
                    CompilerResult(False, errors=["second failure"], failure_kind="lean"),
                    CompilerResult(True),
                ]

            def check_many(self, requests):
                self.assert_single(requests)
                return [self.results.pop(0)]

            @staticmethod
            def assert_single(requests):
                if len(requests) != 1:
                    raise AssertionError("expected one request")

        prover = self.make_prover(max_negation_probe_turns=0)
        prover.max_prove_turns = 3
        model_responses = [
            response([ToolCall("turn-1", "lean_compile", {"proof_body": "by omega"})]),
            response([ToolCall("turn-2", "lean_compile", {"proof_body": "by simp"})]),
            response([ToolCall("turn-3", "lean_compile", {"proof_body": "by trivial"})]),
        ]
        requests = []

        def chat(messages, *_args, **_kwargs):
            requests.append(deepcopy(messages))
            return model_responses[len(requests) - 1]

        with patch.object(prover, "_chat", side_effect=chat):
            result = prover.prove_node(
                SequentialCompiler(),
                "root",
                NODE_DECL,
                "Prove the node.",
                "",
                "import Mathlib",
            )

        self.assertEqual(result.signal, ProofSignal.SOLVED)
        self.assertEqual([message["role"] for message in requests[2]], [
            "system", "user", "assistant", "tool",
        ])
        self.assertEqual(requests[2][2]["tool_calls"][0]["id"], "turn-2")
        self.assertEqual(requests[2][3]["tool_call_id"], "turn-2")
        self.assertNotIn("turn-1", json.dumps(requests[2]))

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
