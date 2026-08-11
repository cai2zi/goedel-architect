from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

from blueprint import _node_hash, _parse_blueprint, _safe_phase2_header  # noqa: E402
from kimina_lean_compiler import CompilerResult  # noqa: E402
from phase1b import (  # noqa: E402
    Phase1BPlan,
    Phase1BValidation,
    _call_editor,
    _commit_assessment,
    _execute_phase1b_mathlib_search,
    _mathlib_search_eligibility,
    _normalized_obligation_signature,
    _stable_gate,
    _two_stage_deterministic_stable,
    run_phase1b_patch_session,
)
from semantic_audit import (  # noqa: E402
    FormalDecompilerResult,
    FormalNodeTranslation,
    StrictComparatorResult,
)
from semantic_fidelity import SemanticIssue  # noqa: E402
from mathlib_retrieval import LemmaResult  # noqa: E402


def _standalone(*issues):
    return SimpleNamespace(
        issues=tuple(issues), skipped_pending_node_count=0,
        checked_node_count=1, cached_node_count=0, duration_ms=0.0,
        not_run_reason="",
    )


def _validation(
    *, lean=True, semantic=(), structural=(), standalone=(), pending=(),
    decompiler=None, comparator=None, obligations=(), audit=True,
):
    return Phase1BValidation(
        CompilerResult(lean, errors=[] if lean else ["lean failure"]),
        list(semantic), [], list(structural), _standalone(*standalone),
        tuple(pending), formal_decompiler_result=decompiler,
        strict_comparator_result=comparator,
        open_semantic_obligations=tuple(obligations),
        semantic_audit_required=audit,
    )


def _decompiler(nodes, *, vacuous=()):
    vacuous = set(vacuous)
    return FormalDecompilerResult(
        tuple(FormalNodeTranslation(
            name, kind, name,
            "vacuous" if name in vacuous else (
                "objectDefinition" if kind == "definition" else "proposition"
            ), (), (),
        ) for name, kind in nodes),
        "", "", "stop", "id", 0, 0, 0,
    )


def _comparator(*, target=True, grounding=True, passed=False, disconnected=0):
    unreachable = tuple({
        "step_id": f"S{index:03d}", "justified_side_branch": False,
        "reason": "disconnected",
    } for index in range(1, disconnected + 1))
    return StrictComparatorResult(
        (), {
            "translation": "root", "target_object_preserved": target,
            "answer_grounded": grounding, "reasons": [],
        }, unreachable, (), (), passed, "", "", "stop", "id", 0, 0, 0,
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
                "action": "replace", "node_name": "setup",
                "expected_node_hash": _node_hash(blueprint.node_by_name("setup")),
                "replacement": _replacement(value),
            }]}),
        ),
    )
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=[call]),
        finish_reason="tool_calls",
    )], usage=None)


def _search_response(*, mixed=False):
    search = SimpleNamespace(
        id="search", type="function",
        function=SimpleNamespace(
            name="mathlib_search",
            arguments=json.dumps({"queries": [{
                "query": "Finset image cardinality",
                "target_node_names": ["setup"],
            }]}),
        ),
    )
    calls = [search]
    if mixed:
        calls.append(SimpleNamespace(
            id="early-edit", type="function",
            function=SimpleNamespace(name="editBlueprintSubgraph", arguments="{}"),
        ))
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=calls),
        finish_reason="tool_calls",
    )], usage=None)


class _FakeRetrieval:
    calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def search(self, query, k):
        type(self).calls += 1
        return [LemmaResult("Finset.card_image_iff", "...", "doc")][:k]


class Phase1BV11MathlibSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeRetrieval.calls = 0

    def test_two_stage_stable_result_preserves_commit_assessment_contract(self) -> None:
        before = {
            "count": 2, "semanticErrors": ["old"], "structuralErrors": [],
            "standaloneErrors": [], "pendingNodes": ["root"], "leanSuccess": False,
        }
        after = {
            "count": 1, "semanticErrors": ["old"], "structuralErrors": [],
            "standaloneErrors": [], "pendingNodes": [], "leanSuccess": False,
        }
        result = _two_stage_deterministic_stable(before, after)
        self.assertTrue(result["passed"])
        self.assertTrue(result["deterministicDebtDecreased"])

    def test_canonical_header_preserves_safe_commands_after_pending_helper(self) -> None:
        header = _safe_phase2_header(
            "import Mathlib\nimport Architect\n\n"
            "def PendingBlueprintClaim (_nodeId : String) : Prop := True\n\n"
            "open Classical\nset_option pp.universes false\n\n"
            "@[blueprint] theorem root : True := by trivial\n"
        )
        self.assertIn("open Classical", header)
        self.assertIn("set_option pp.universes false", header)
        self.assertNotIn("PendingBlueprintClaim", header)

    def test_batched_search_validates_inventory_bounds_and_caches(self) -> None:
        blueprint = _blueprint()
        call = _search_response().choices[0].message.tool_calls[0]
        cache = {}
        with patch("phase1b.MathlibRetrieval", _FakeRetrieval):
            first = _execute_phase1b_mathlib_search(
                [call], blueprint=blueprint, plan=None,
                max_queries=3, max_results=5, cache=cache,
                tracer=None, thm_name="sample", round_index=1, attempt=1,
            )
            second = _execute_phase1b_mathlib_search(
                [call], blueprint=blueprint, plan=None,
                max_queries=3, max_results=5, cache=cache,
                tracer=None, thm_name="sample", round_index=1, attempt=2,
            )
        self.assertEqual(_FakeRetrieval.calls, 1)
        self.assertEqual(first[0]["results"][0]["name"], "Finset.card_image_iff")
        self.assertFalse(first[0]["cache_hit"])
        self.assertTrue(second[0]["cache_hit"])

    def test_search_then_forces_edit_only_and_persists_current_turn_results(self) -> None:
        blueprint = _blueprint()
        edit = _editor_response(blueprint, 1)
        state = {"used": False, "results": []}
        with (
            patch("phase1b.MathlibRetrieval", _FakeRetrieval),
            patch("phase1b.chat_completion_with_retry", side_effect=[
                _search_response(mixed=True), edit,
            ]) as completion,
        ):
            response = _call_editor(
                object(), "model", blueprint=blueprint, plan=None,
                informal_statement="p", prompt_proof="steps", claimed_answer="1",
                validation=_validation(audit=False), retry_feedback={
                    "deterministic_diagnostics": {
                        "lean_errors": ["Unknown identifier `Finset.card_image_iff`"],
                        "standalone_errors": [],
                    },
                },
                round_index=1, attempt=1, tracer=None, thm_name="sample",
                search_state=state, search_cache={}, search_max_queries=3,
                search_max_results=5,
            )
        self.assertIs(response, edit)
        self.assertTrue(state["used"])
        self.assertEqual(completion.call_count, 2)
        self.assertEqual(
            [tool["function"]["name"] for tool in completion.call_args_list[1].kwargs["tools"]],
            ["editBlueprintSubgraph"],
        )
        second_payload = json.loads(completion.call_args_list[1].kwargs["messages"][1]["content"])
        self.assertEqual(second_payload["mode"], "EDIT ONLY")
        self.assertEqual(
            second_payload["mathlib_search"]["results_for_this_turn"][0]["results"][0]["name"],
            "Finset.card_image_iff",
        )

    def test_v13_editor_thinking_sampling_is_scoped_to_editor_request(self) -> None:
        blueprint = _blueprint()
        response = _editor_response(blueprint, 1)
        with patch("phase1b.chat_completion_with_retry", return_value=response) as completion:
            _call_editor(
                object(), "model", blueprint=blueprint, plan=None,
                informal_statement="p", prompt_proof="steps", claimed_answer="1",
                validation=_validation(audit=False), retry_feedback=None,
                round_index=2, attempt=1, tracer=None, thm_name="sample",
                enable_thinking=True, temperature=0.6, top_p=0.95,
                top_k=20, min_p=0.0, presence_penalty=0.0,
                repetition_penalty=1.0, max_tokens=16384,
            )
        kwargs = completion.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.6)
        self.assertEqual(kwargs["top_p"], 0.95)
        self.assertEqual(kwargs["max_completion_tokens"], 16384)
        self.assertTrue(kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(kwargs["extra_body"]["top_k"], 20)
        self.assertIsInstance(kwargs["seed"], int)

    def test_search_is_closed_without_specific_lean_api_error(self) -> None:
        blueprint = _blueprint()
        edit = _editor_response(blueprint, 1)
        state = {"used": False, "results": []}
        with patch("phase1b.chat_completion_with_retry", return_value=edit) as completion:
            response = _call_editor(
                object(), "model", blueprint=blueprint, plan=None,
                informal_statement="p", prompt_proof="steps", claimed_answer="1",
                validation=_validation(audit=False), retry_feedback={
                    "kind": "semanticProgressGate",
                    "errors": ["rootTargetObject"],
                },
                round_index=1, attempt=1, tracer=None, thm_name="sample",
                search_state=state, search_cache={}, search_max_queries=3,
                search_max_results=5,
            )
        self.assertIs(response, edit)
        self.assertFalse(state["used"])
        self.assertEqual(
            [tool["function"]["name"] for tool in completion.call_args.kwargs["tools"]],
            ["editBlueprintSubgraph"],
        )

    def test_search_eligibility_excludes_blueprint_dependency_and_plain_mismatch(self) -> None:
        validation = _validation(audit=False)
        feedback = {"deterministic_diagnostics": {
            "lean_errors": [
                "Unknown identifier `setup`",
                "application type mismatch",
            ],
            "standalone_errors": [{
                "errorKind": "unknownIdentifier",
                "identifiers": ["setup"],
                "diagnostic": "Unknown identifier `setup`",
            }],
        }}
        result = _mathlib_search_eligibility(
            validation, retry_feedback=feedback,
            blueprint_node_names=["setup", "root"], policy="leanErrorsOnly",
        )
        self.assertFalse(result["eligible"])

    def test_search_eligibility_accepts_external_api_and_synthesis_errors(self) -> None:
        validation = _validation(audit=False)
        feedback = {"deterministic_diagnostics": {
            "lean_errors": [
                "Unknown constant `Finset.card_image_iff`",
                "failed to synthesize Fintype α",
            ],
            "standalone_errors": [],
        }}
        result = _mathlib_search_eligibility(
            validation, retry_feedback=feedback,
            blueprint_node_names=["setup", "root"], policy="leanErrorsOnly",
        )
        self.assertTrue(result["eligible"])
        self.assertEqual(len(result["reasons"]), 2)

    def test_invalid_or_excess_queries_return_protocol_diagnostic_not_infra(self) -> None:
        blueprint = _blueprint()
        call = SimpleNamespace(function=SimpleNamespace(
            arguments=json.dumps({"queries": [
                {"query": str(i), "target_node_names": ["setup"]}
                for i in range(4)
            ]}),
        ))
        rows = _execute_phase1b_mathlib_search(
            [call], blueprint=blueprint, plan=None,
            max_queries=3, max_results=5, cache={}, tracer=None,
            thm_name="sample", round_index=1, attempt=1,
        )
        self.assertIn("search queries must contain 1..3 items", rows[0]["protocol_errors"])
        self.assertEqual(_FakeRetrieval.calls, 0)


class Phase1BV10StableGateTest(unittest.TestCase):
    def test_rejects_new_lean_static_and_changed_standalone_failures(self) -> None:
        baseline = _validation(audit=False)
        static = SemanticIssue("reflexiveStep", "bad", node_name="setup", step_id="S001")
        standalone_issue = SimpleNamespace(to_dict=lambda: {
            "code": "phase2StandaloneFailed", "nodeName": "setup",
            "errorKind": "unknownIdentifier", "preflightHash": "x",
        })
        candidate = _validation(
            lean=False, semantic=(static,), standalone=(standalone_issue,), audit=False,
        )
        result = _stable_gate(
            baseline, candidate,
            changed_nodes=("setup",),
        )
        self.assertFalse(result["passed"])
        self.assertIn("wholeFileLeanFailed", result["errors"])
        self.assertTrue(any("newSemanticErrors" in item for item in result["errors"]))
        self.assertTrue(any("changedConcreteStandaloneFailed" in item for item in result["errors"]))

    def test_existing_debt_may_remain_and_pending_may_decrease(self) -> None:
        issue = SemanticIssue("vacuousTrueStep", "old", node_name="setup", step_id="S001")
        baseline = _validation(semantic=(issue,), pending=("setup",), audit=False)
        candidate = _validation(semantic=(issue,), pending=(), audit=False)
        result = _stable_gate(
            baseline, candidate,
            changed_nodes=("setup",),
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["deterministicDebtDecreased"])


class Phase1BV10CommitAssessmentTest(unittest.TestCase):
    @staticmethod
    def _foundation_pair():
        baseline = _parse_blueprint('''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001")
  (statement := /-- root -/) (proof := /-- root -/)]
theorem root : (1 : Nat) = 1 := by sorry_using []
''', "root")
        candidate = _parse_blueprint('''import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001")
  (statement := /-- shared -/)]
def shared_object : Nat := 1
@[blueprint (title := "COT_STEP:S001")
  (statement := /-- root -/) (proof := /-- root -/)]
theorem root : shared_object = 1 := by sorry_using []
''', "root")
        return baseline, candidate

    def test_integrated_foundation_allowed_early_but_not_in_closure(self) -> None:
        baseline_bp, candidate_bp = self._foundation_pair()
        baseline = _validation(
            decompiler=_decompiler((("root", "theorem"),)),
            comparator=_comparator(target=True, grounding=True),
        )
        candidate = _validation(
            decompiler=_decompiler((
                ("shared_object", "definition"), ("root", "theorem"),
            )), comparator=_comparator(target=True, grounding=True),
        )
        stable = {"passed": True, "errors": [], "deterministicDebtDecreased": False,
                  "baselineDebt": {"count": 0}, "candidateDebt": {"count": 0}}
        hard = {"effectiveNodes": ["shared_object", "root"], "noOpNodes": [],
                "edits": [{"action": "add", "nodeName": "shared_object"},
                          {"action": "replace", "nodeName": "root"}]}
        early = _commit_assessment(
            baseline_bp, candidate_bp, baseline, candidate, stable=stable,
            hard_result=hard, plan=None, semantic_manifest=None,
            closure_mode=False, foundation_debt_open=False,
        )
        self.assertTrue(early["passed"])
        self.assertTrue(early["foundationOnly"])
        self.assertEqual(early["foundationConsumers"]["shared_object"], ["root"])
        closure = _commit_assessment(
            baseline_bp, candidate_bp, baseline, candidate, stable=stable,
            hard_result=hard, plan=None, semantic_manifest=None,
            closure_mode=True, foundation_debt_open=False,
        )
        self.assertFalse(closure["passed"])
        self.assertIn("foundationOnlyNotAllowedInClosure", closure["errors"])

    def test_consecutive_foundation_only_is_rejected(self) -> None:
        baseline_bp, candidate_bp = self._foundation_pair()
        validation = _validation(
            decompiler=_decompiler((("root", "theorem"),)),
            comparator=_comparator(target=True, grounding=True),
        )
        candidate = _validation(
            decompiler=_decompiler((
                ("shared_object", "definition"), ("root", "theorem"),
            )), comparator=_comparator(target=True, grounding=True),
        )
        stable = {"passed": True, "errors": [], "deterministicDebtDecreased": False,
                  "baselineDebt": {"count": 0}, "candidateDebt": {"count": 0}}
        hard = {"effectiveNodes": ["shared_object", "root"], "noOpNodes": [],
                "edits": [{"action": "add", "nodeName": "shared_object"},
                          {"action": "replace", "nodeName": "root"}]}
        result = _commit_assessment(
            baseline_bp, candidate_bp, validation, candidate, stable=stable,
            hard_result=hard, plan=None, semantic_manifest=None,
            closure_mode=False, foundation_debt_open=True,
        )
        self.assertFalse(result["passed"])
        self.assertIn("consecutiveFoundationOnlyCommit", result["errors"])

    def test_closure_allows_added_object_only_when_object_obligation_resolves(self) -> None:
        baseline_bp, candidate_bp = self._foundation_pair()
        obligation = {
            "obligation_id": "semantic:rootTargetObject:x",
            "category": "rootTargetObject", "step_id": "",
            "requirement": "preserve root object",
        }
        baseline = _validation(
            decompiler=_decompiler((("root", "theorem"),)),
            comparator=_comparator(target=False, grounding=True),
            obligations=(obligation,),
        )
        candidate = _validation(
            decompiler=_decompiler((
                ("shared_object", "definition"), ("root", "theorem"),
            )), comparator=_comparator(target=True, grounding=True),
        )
        stable = {"passed": True, "errors": [], "deterministicDebtDecreased": False,
                  "baselineDebt": {"count": 0}, "candidateDebt": {"count": 0}}
        hard = {"effectiveNodes": ["shared_object", "root"], "noOpNodes": [],
                "edits": [{"action": "add", "nodeName": "shared_object"},
                          {"action": "replace", "nodeName": "root"}]}
        result = _commit_assessment(
            baseline_bp, candidate_bp, baseline, candidate, stable=stable,
            hard_result=hard, plan=None, semantic_manifest=None,
            closure_mode=True, foundation_debt_open=False,
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["closureAuxiliaryAllowed"])

    def test_obligation_relocation_uses_source_step_hash(self) -> None:
        manifest = SimpleNamespace(by_id={
            "S003": SimpleNamespace(source_sha256="source-hash"),
        })
        left = {"category": "missing_clauses", "step_id": "S003",
                "node_names": ["old"], "requirement": "old wording"}
        right = {"category": "missing_clauses", "step_id": "S003.detail",
                 "node_names": ["new"], "requirement": "new wording"}
        self.assertEqual(
            _normalized_obligation_signature(left, manifest),
            _normalized_obligation_signature(right, manifest),
        )


class Phase1BV10SchedulingTest(unittest.TestCase):
    def _run(self, strategy: str, *, editor_side_effect, planner=None):
        blueprint = _blueprint()
        history: list[str] = []
        labels: list[str] = []

        def validation(candidate, **_kwargs):
            pending = ': PendingBlueprintClaim "setup"' in candidate.lean_file
            wrong = ": x = 2 :=" in candidate.lean_file
            return _validation(
                lean=not wrong, pending=("setup",) if pending else (), audit=False,
            )

        patches = [
            patch("phase1b.validate_candidate", side_effect=validation),
            patch("phase1b._call_editor", side_effect=editor_side_effect),
        ]
        if planner is not None:
            patches.append(patch("phase1b.run_planner", side_effect=planner))
        entered = []
        try:
            for item in patches:
                entered.append(item.start())
            result = run_phase1b_patch_session(
                object(), "model", blueprint, compiler=SimpleNamespace(),
                informal_statement="p", prompt_proof="S001 then S002",
                claimed_answer="1", semantic_manifest=None,
                semantic_fidelity_enabled=False, semantic_require_step_ids=False,
                semantic_static_gate=False, max_rounds=2,
                phase2_contract_check_concurrency=1, tracer=None,
                thm_name="sample", candidate_history=history,
                candidate_labels=labels, repair_strategy=strategy,
                editor_attempts_per_turn=3,
            )
        finally:
            for item in reversed(patches):
                item.stop()
        return result, entered, labels

    def test_direct_edit_retries_stable_failure_from_same_baseline(self) -> None:
        blueprint = _blueprint()
        result, entered, labels = self._run(
            "directEdit", editor_side_effect=[
                _editor_response(blueprint, 2), _editor_response(blueprint, 1),
            ],
        )
        editor = entered[1]
        self.assertEqual(editor.call_count, 2)
        self.assertEqual(
            editor.call_args_list[0].kwargs["blueprint"].lean_file,
            editor.call_args_list[1].kwargs["blueprint"].lean_file,
        )
        self.assertIn(": x = 1 :=", result.lean_file)
        self.assertEqual(labels[-1], "phase1b_final")

    def test_plan_direct_plans_once_for_stable_retry(self) -> None:
        blueprint = _blueprint()
        plan = Phase1BPlan(("none",), ("setup",), (), "make setup concrete")
        result, entered, _labels = self._run(
            "planDirect", planner=[plan], editor_side_effect=[
                _editor_response(blueprint, 2), _editor_response(blueprint, 1),
            ],
        )
        self.assertEqual(entered[2].call_count, 1)
        self.assertEqual(entered[1].call_count, 2)
        self.assertIn(": x = 1 :=", result.lean_file)


if __name__ == "__main__":
    unittest.main()
