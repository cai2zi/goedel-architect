from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

from blueprint import _run_phase1_tool_session  # noqa: E402
from kimina_lean_compiler import CompilerResult  # noqa: E402
from semantic_fidelity import CotManifest, CotStep  # noqa: E402


class FakeCall:
    def __init__(self, call_id: str, name: str, args: dict) -> None:
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=json.dumps(args))

    def model_dump(self) -> dict:
        return {
            "id": self.id, "type": "function",
            "function": {"name": self.function.name,
                         "arguments": self.function.arguments},
        }


def response(*calls: FakeCall):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=list(calls)),
            finish_reason="tool_calls",
        )],
        usage=None,
    )


class Phase1ToolProtocolTest(unittest.TestCase):
    def test_last_turn_requires_compile_and_returns_exact_tool_candidate(self) -> None:
        code = '''import Mathlib
import Architect
def PendingBlueprintClaim (_nodeId : String) : Prop := True
@[blueprint (statement := /-- root -/) (proof := /-- root -/)]
theorem root : True := by sorry_using []
'''
        requests = []

        def complete(_client, **kwargs):
            requests.append(kwargs)
            return response(FakeCall("c1", "lean_compile", {"lean_code": code}))

        compiler = SimpleNamespace(
            check_blueprint=lambda candidate, target: CompilerResult(candidate == code and target == "root")
        )
        with patch("blueprint.chat_completion_with_retry", side_effect=complete):
            result = _run_phase1_tool_session(
                object(), "model", ({"role": "system", "content": "s"},),
                compiler=compiler, target_name="root",
                retrieval=SimpleNamespace(search=lambda *_args: []), tracer=None,
                thm_name="sample", attempt=1, max_tool_turns=1,
                max_tool_calls_per_turn=3, mathlib_search_max_calls=3,
                tool_cache={}, search_state={"count": 0},
                enable_thinking=True, temperature=0.6, top_p=0.95,
                top_k=20, min_p=0.0, presence_penalty=0.0,
                repetition_penalty=1.0, max_tokens=16384,
            )
        self.assertEqual(result.successful_lean_code, code)
        self.assertEqual(requests[0]["tool_choice"], "required")
        self.assertEqual([tool["function"]["name"] for tool in requests[0]["tools"]], [
            "lean_compile",
        ])
        self.assertEqual(requests[0]["temperature"], 0.6)
        self.assertEqual(requests[0]["top_p"], 0.95)
        self.assertTrue(
            requests[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        )
        self.assertEqual(requests[0]["extra_body"]["top_k"], 20)
        self.assertIsInstance(requests[0]["seed"], int)

    def test_search_budget_is_shared_and_latest_exchange_is_bounded(self) -> None:
        code = '''import Mathlib
import Architect
def PendingBlueprintClaim (_nodeId : String) : Prop := True
@[blueprint (statement := /-- root -/) (proof := /-- root -/)]
theorem root : True := by sorry_using []
'''
        responses = [
            response(FakeCall("s1", "mathlib_search", {"query": "Nat addition", "k": 3})),
            response(FakeCall("c1", "lean_compile", {"lean_code": code})),
        ]
        messages_seen = []

        def complete(_client, **kwargs):
            messages_seen.append(kwargs["messages"])
            return responses[len(messages_seen) - 1]

        search_state = {"count": 0}
        with patch("blueprint.chat_completion_with_retry", side_effect=complete):
            result = _run_phase1_tool_session(
                object(), "model", (
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                ), compiler=SimpleNamespace(
                    check_blueprint=lambda *_args: CompilerResult(True)
                ), target_name="root", retrieval=SimpleNamespace(search=lambda *_args: []),
                tracer=None, thm_name="sample", attempt=1, max_tool_turns=2,
                max_tool_calls_per_turn=3, mathlib_search_max_calls=1,
                tool_cache={}, search_state=search_state,
            )
        self.assertEqual(result.successful_lean_code, code)
        self.assertEqual(search_state["count"], 1)
        self.assertEqual([message["role"] for message in messages_seen[1]], [
            "system", "user", "assistant", "tool",
        ])

    def test_repairable_semantic_error_is_deferred_to_phase1b(self) -> None:
        bad = '''import Mathlib
import Architect
def PendingBlueprintClaim (_nodeId : String) : Prop := True
@[blueprint (title := "COT_STEP:S001")
  (statement := /-- The final answer is represented by a true proposition. -/)
  (proof := /-- This deliberately exercises the semantic rejection path. -/)]
theorem root : True := by sorry_using []
'''
        responses = [response(FakeCall("bad", "lean_compile", {"lean_code": bad}))]
        messages_seen = []

        def complete(_client, **kwargs):
            messages_seen.append(kwargs["messages"])
            return responses[len(messages_seen) - 1]

        source_text = "Therefore the answer is one."
        semantic_manifest = CotManifest((CotStep(
            "S001", 0, len(source_text), source_text, "hash",
        ),))
        compiler = SimpleNamespace(check_blueprint=lambda *_args: CompilerResult(True))
        with patch("blueprint.chat_completion_with_retry", side_effect=complete):
            result = _run_phase1_tool_session(
                object(), "model", (
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": f"[COT_STEP S001]{source_text}"},
                ), compiler=compiler, target_name="root",
                retrieval=SimpleNamespace(search=lambda *_args: []), tracer=None,
                thm_name="sample", attempt=1, max_tool_turns=2,
                max_tool_calls_per_turn=3, mathlib_search_max_calls=3,
                tool_cache={}, search_state={"count": 0},
                semantic_manifest=semantic_manifest, claimed_answer="1",
                semantic_fidelity_enabled=True, semantic_require_step_ids=True,
                semantic_static_gate=True,
            )

        self.assertEqual(result.successful_lean_code, bad)
        self.assertEqual(len(messages_seen), 1)

    def test_immutable_binding_error_continues_phase1a_tool_loop(self) -> None:
        bad = '''import Mathlib
import Architect
def PendingBlueprintClaim (_nodeId : String) : Prop := True
@[blueprint (title := "COT_STEP:S999")
  (statement := /-- An incorrectly bound final claim. -/)
  (proof := /-- Preserve the claim. -/)]
theorem root : (2 : Nat) - 1 = 1 := by sorry_using []
'''
        good = '''import Mathlib
import Architect
def PendingBlueprintClaim (_nodeId : String) : Prop := True
@[blueprint (title := "COT_STEP:S001")
  (statement := /-- Subtracting one from two gives one. -/)
  (proof := /-- Evaluate the subtraction. -/)]
theorem root : (2 : Nat) - 1 = 1 := by sorry_using []
'''
        responses = [
            response(FakeCall("bad", "lean_compile", {"lean_code": bad})),
            response(FakeCall("good", "lean_compile", {"lean_code": good})),
        ]
        messages_seen = []

        def complete(_client, **kwargs):
            messages_seen.append(kwargs["messages"])
            return responses[len(messages_seen) - 1]

        source_text = "Therefore the answer is one."
        semantic_manifest = CotManifest((CotStep(
            "S001", 0, len(source_text), source_text, "hash",
        ),))
        with patch("blueprint.chat_completion_with_retry", side_effect=complete):
            result = _run_phase1_tool_session(
                object(), "model", ({"role": "system", "content": "s"},),
                compiler=SimpleNamespace(check_blueprint=lambda *_args: CompilerResult(True)),
                target_name="root", retrieval=SimpleNamespace(search=lambda *_args: []),
                tracer=None, thm_name="sample", attempt=1, max_tool_turns=2,
                max_tool_calls_per_turn=3, mathlib_search_max_calls=3,
                tool_cache={}, search_state={"count": 0},
                semantic_manifest=semantic_manifest, claimed_answer="1",
                semantic_fidelity_enabled=True, semantic_require_step_ids=True,
                semantic_static_gate=True,
            )
        self.assertEqual(result.successful_lean_code, good)
        self.assertEqual(len(messages_seen), 2)
        self.assertIn("semantic-fidelity validation FAILED", messages_seen[1][-1]["content"])
        self.assertIn("unknownStepMapping", messages_seen[1][-1]["content"])

    def test_semantic_warning_does_not_block_success(self) -> None:
        code = '''import Mathlib
import Architect
def PendingBlueprintClaim (_nodeId : String) : Prop := True
@[blueprint (title := "COT_STEP:S001")
  (statement := /-- Binds: none. Assumes: none. Claims: two is greater than one. Use: none. -/)
  (proof := /-- Derive: arithmetic. -/)]
lemma orphan : (2 : Nat) > 1 := by sorry_using []
@[blueprint (title := "COT_STEP:S001")
  (statement := /-- Subtracting one from two gives the claimed answer one. -/)
  (proof := /-- Evaluate the subtraction. -/)]
theorem root : (2 : Nat) - 1 = 1 := by sorry_using []
'''
        source_text = "Therefore the answer is one."
        semantic_manifest = CotManifest((CotStep(
            "S001", 0, len(source_text), source_text, "hash",
        ),))
        with patch(
            "blueprint.chat_completion_with_retry",
            return_value=response(FakeCall("compile", "lean_compile", {"lean_code": code})),
        ) as complete:
            result = _run_phase1_tool_session(
                object(), "model", ({"role": "system", "content": "s"},),
                compiler=SimpleNamespace(
                    check_blueprint=lambda *_args: CompilerResult(True)
                ), target_name="root", retrieval=SimpleNamespace(search=lambda *_args: []),
                tracer=None, thm_name="sample", attempt=1, max_tool_turns=2,
                max_tool_calls_per_turn=3, mathlib_search_max_calls=3,
                tool_cache={}, search_state={"count": 0},
                semantic_manifest=semantic_manifest, claimed_answer="1",
                semantic_fidelity_enabled=True, semantic_require_step_ids=True,
                semantic_static_gate=True,
            )
        self.assertEqual(result.successful_lean_code, code)
        self.assertEqual(complete.call_count, 1)


if __name__ == "__main__":
    unittest.main()
