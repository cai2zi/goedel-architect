from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

from blueprint import _node_hash, _parse_blueprint  # noqa: E402
from kimina_lean_compiler import CompilerResult  # noqa: E402
from phase1b import (  # noqa: E402
    Phase1BPlan,
    Phase1BValidation,
    ProgressDecision,
    run_phase1b_patch_session,
)


def _blueprint():
    return _parse_blueprint('''import Mathlib
import Architect
def PendingBlueprintClaim (_nodeId : String) : Prop := True
@[blueprint (title := "COT_STEP:S001")
  (statement := /-- source relation -/) (proof := /-- formalize it -/)]
lemma setup (x : Nat) (h : x = 1) : PendingBlueprintClaim "setup" := by sorry_using []
@[blueprint (title := "COT_STEP:S002")
  (statement := /-- final result -/) (proof := /-- use setup -/)]
theorem root : (1 : Nat) = 1 := by sorry_using [setup]
''', "root")


def _replacement(value: int) -> str:
    return f'''@[blueprint (title := "COT_STEP:S001")
  (statement := /-- source relation -/) (proof := /-- formalize it -/)]
lemma setup (x : Nat) (h : x = 1) : x = {value} := by sorry_using []'''


def _editor_response(blueprint, value: int):
    call = SimpleNamespace(
        id=f"edit-{value}", type="function",
        function=SimpleNamespace(
            name="editBlueprintSubgraph",
            arguments=json.dumps({"edits": [{
                "action": "replace",
                "node_name": "setup",
                "expected_node_hash": _node_hash(blueprint.node_by_name("setup")),
                "replacement": _replacement(value),
            }]}),
        ),
    )
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=[call]),
        finish_reason="tool_calls",
    )], usage=None)


def _validation(blueprint, **_kwargs):
    pending = ': PendingBlueprintClaim "setup"' in blueprint.lean_file
    wrong = ": x = 2 :=" in blueprint.lean_file
    success = not pending and not wrong
    standalone = SimpleNamespace(
        issues=(), skipped_pending_node_count=0, checked_node_count=1,
        cached_node_count=0, duration_ms=0.0, not_run_reason="",
    )
    return Phase1BValidation(
        CompilerResult(success, errors=[] if success else ["soft failure"]),
        [], [], [], standalone,
        ("setup",) if pending else (),
        semantic_audit_required=False,
    )


class Phase1BV9SchedulingTest(unittest.TestCase):
    def _run(self, strategy: str, *, editor_side_effect, planner=None, controller=None):
        blueprint = _blueprint()
        history: list[str] = []
        labels: list[str] = []
        patches = [
            patch("phase1b.validate_candidate", side_effect=_validation),
            patch("phase1b._call_editor", side_effect=editor_side_effect),
        ]
        if planner is not None:
            patches.append(patch("phase1b.run_planner", side_effect=planner))
        if controller is not None:
            patches.append(patch("phase1b.run_progress_controller", side_effect=controller))
        entered = []
        try:
            for item in patches:
                entered.append(item.start())
            result = run_phase1b_patch_session(
                object(), "model", blueprint,
                compiler=SimpleNamespace(), informal_statement="p",
                prompt_proof="S001 then S002", claimed_answer="1",
                semantic_manifest=None, semantic_fidelity_enabled=False,
                semantic_require_step_ids=False, semantic_static_gate=False,
                max_rounds=2, phase2_contract_check_concurrency=1,
                tracer=None, thm_name="sample", candidate_history=history,
                candidate_labels=labels, repair_strategy=strategy,
                editor_attempts_per_turn=3,
            )
        finally:
            for item in reversed(patches):
                item.stop()
        return result, entered, history, labels

    def test_direct_edit_skips_planner_and_commits_first_hard_valid_candidate(self) -> None:
        blueprint = _blueprint()
        with patch("phase1b.run_planner") as planner, patch(
            "phase1b.run_progress_controller"
        ) as controller:
            result, entered, _history, labels = self._run(
                "directEdit", editor_side_effect=[_editor_response(blueprint, 1)],
            )
        self.assertIn(": x = 1 :=", result.lean_file)
        planner.assert_not_called()
        controller.assert_not_called()
        self.assertEqual(entered[1].call_count, 1)
        self.assertEqual(labels, ["phase1b_round_1_attempt_1", "phase1b_final"])

    def test_plan_direct_plans_once_and_does_not_call_controller(self) -> None:
        blueprint = _blueprint()
        plan = Phase1BPlan(("none",), ("setup",), (), "make setup concrete")
        with patch("phase1b.run_progress_controller") as controller:
            result, entered, _history, _labels = self._run(
                "planDirect", planner=[plan],
                editor_side_effect=[_editor_response(blueprint, 1)],
            )
        self.assertIn(": x = 1 :=", result.lean_file)
        self.assertEqual(entered[2].call_count, 1)
        controller.assert_not_called()

    def test_controller_retry_uses_same_turn_baseline_and_same_plan(self) -> None:
        blueprint = _blueprint()
        plan = Phase1BPlan(("none",), ("setup",), (), "make setup concrete")
        result, entered, _history, _labels = self._run(
            "progressController", planner=[plan],
            editor_side_effect=[
                _editor_response(blueprint, 2),
                _editor_response(blueprint, 1),
            ],
            controller=[ProgressDecision("RETRY_EDIT", "x = 2 misses the source relation")],
        )
        editor = entered[1]
        self.assertEqual(editor.call_count, 2)
        first_baseline = editor.call_args_list[0].kwargs["blueprint"].lean_file
        second_baseline = editor.call_args_list[1].kwargs["blueprint"].lean_file
        self.assertEqual(first_baseline, blueprint.lean_file)
        self.assertEqual(second_baseline, blueprint.lean_file)
        self.assertIs(editor.call_args_list[0].kwargs["plan"], plan)
        self.assertIs(editor.call_args_list[1].kwargs["plan"], plan)
        self.assertIn(": x = 1 :=", result.lean_file)

    def test_hard_failure_retries_within_turn_without_replanning(self) -> None:
        blueprint = _blueprint()
        plan = Phase1BPlan(("none",), ("setup",), (), "make setup concrete")
        missing_call = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=[]),
            finish_reason="stop",
        )], usage=None)
        result, entered, _history, _labels = self._run(
            "planDirect", planner=[plan],
            editor_side_effect=[missing_call, _editor_response(blueprint, 1)],
        )
        self.assertEqual(entered[2].call_count, 1)
        self.assertEqual(entered[1].call_count, 2)
        self.assertEqual(
            entered[1].call_args_list[1].kwargs["blueprint"].lean_file,
            blueprint.lean_file,
        )
        self.assertIn(": x = 1 :=", result.lean_file)

    def test_plan_direct_commits_soft_invalid_candidate_as_next_turn_baseline(self) -> None:
        plan = Phase1BPlan(("none",), ("setup",), (), "make setup concrete")
        values = iter((2, 1))

        def editor(*_args, **kwargs):
            return _editor_response(kwargs["blueprint"], next(values))

        result, entered, _history, _labels = self._run(
            "planDirect", planner=[plan, plan], editor_side_effect=editor,
        )
        editor_mock = entered[1]
        self.assertEqual(editor_mock.call_count, 2)
        self.assertIn(": x = 2 :=", editor_mock.call_args_list[1].kwargs["blueprint"].lean_file)
        self.assertIn(": x = 1 :=", result.lean_file)


if __name__ == "__main__":
    unittest.main()
