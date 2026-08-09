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
        code = "import Mathlib\nimport Architect\ntheorem root : True := by trivial"
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
            )
        self.assertEqual(result.successful_lean_code, code)
        self.assertEqual(requests[0]["tool_choice"], "required")
        self.assertEqual([tool["function"]["name"] for tool in requests[0]["tools"]], [
            "lean_compile",
        ])

    def test_search_budget_is_shared_and_latest_exchange_is_bounded(self) -> None:
        code = "import Mathlib\nimport Architect\ntheorem root : True := by trivial"
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

    def test_semantic_error_continues_tool_loop_after_lean_success(self) -> None:
        bad = '''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001")]
theorem root : True := by sorry_using []
'''
        good = '''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001")]
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
            "S001", 0, len(source_text), source_text, "hash", role="conclusion",
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

        self.assertEqual(result.successful_lean_code, good)
        self.assertEqual(len(messages_seen), 2)
        first_tool_feedback = messages_seen[1][-1]["content"]
        self.assertIn("semantic-fidelity validation FAILED", first_tool_feedback)
        self.assertIn("vacuousTrueRoot", first_tool_feedback)
        self.assertIn("S001/root", first_tool_feedback)
        self.assertNotIn(source_text, first_tool_feedback)

    def test_semantic_warning_does_not_block_success(self) -> None:
        code = '''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001")]
lemma orphan : (2 : Nat) > 1 := by sorry_using []
@[blueprint (title := "COT_STEP:S001")]
theorem root : (2 : Nat) - 1 = 1 := by sorry_using []
'''
        source_text = "Therefore the answer is one."
        semantic_manifest = CotManifest((CotStep(
            "S001", 0, len(source_text), source_text, "hash", role="conclusion",
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
