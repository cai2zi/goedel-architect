from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

from blueprint import (  # noqa: E402
    _parse_blueprint,
    _phase1a_contract_errors,
    _phase1b_search_or_edit_response,
    _phase2_preflight_file,
    _update_semantic_obligations,
)
from cot_blueprint_refine.formal_steps import make_formal_step_manifest  # noqa: E402
from mathlib_retrieval import LemmaResult  # noqa: E402
from semantic_audit import (  # noqa: E402
    FormalDecompilerResult,
    SemanticAuditFormatError,
    StrictComparatorResult,
    build_formal_view,
    formal_decompiler_messages,
    parse_formal_decompiler,
    parse_strict_comparator,
    run_formal_decompiler,
    strict_comparator_messages,
)
from semantic_fidelity import parse_cot_manifest  # noqa: E402


def _fixture():
    source = "Let x be one.\nTherefore x equals one."
    manifest = parse_cot_manifest(make_formal_step_manifest(
        source, [(0, 14), (14, len(source))],
    ))
    blueprint = _parse_blueprint('''import Mathlib
import Architect
/- hidden block prose -/
@[blueprint (title := "COT_STEP:S001")
  (statement := /-- metadata must not reach decompiler -/)
  (proof := /-- proof prose must not reach decompiler -/)]
def sourceValue : Nat := 1 -- hidden line prose
@[blueprint (title := "COT_STEP:S001")
  (statement := /-- metadata -/) (proof := /-- prose -/)]
lemma setup : sourceValue = 1 := by sorry_using [sourceValue]
@[blueprint (title := "COT_STEP:S002")
  (statement := /-- final metadata -/) (proof := /-- final prose -/)]
theorem root : sourceValue = 1 := by sorry_using [setup]
''', "root")
    return manifest, blueprint


def _decompiler_json(*, vacuous: str = "") -> str:
    return json.dumps({"nodes": [
        {"node_name": "sourceValue", "kind": "definition",
         "translation": "Defines sourceValue as one.",
         "semantic_effect": "objectDefinition", "introduced_objects": ["sourceValue"],
         "referenced_objects": []},
        {"node_name": "setup", "kind": "lemma", "translation": "sourceValue equals one.",
         "semantic_effect": "vacuous" if vacuous == "setup" else "proposition",
         "introduced_objects": [], "referenced_objects": ["sourceValue"]},
        {"node_name": "root", "kind": "theorem", "translation": "sourceValue equals one.",
         "semantic_effect": "proposition", "introduced_objects": [],
         "referenced_objects": ["sourceValue"]},
    ]})


def _decompiler_result(view, *, vacuous: str = "") -> FormalDecompilerResult:
    nodes = parse_formal_decompiler(_decompiler_json(vacuous=vacuous), view=view)
    return FormalDecompilerResult(nodes, _decompiler_json(vacuous=vacuous), "", "stop", "r", 1, 1, 2)


def _comparator_json(*, missing: bool = False, obligations=()) -> str:
    issue = [{"clause": "x is introduced", "node_names": ["sourceValue"],
              "reason": "The object binding is absent."}] if missing else []
    return json.dumps({
        "steps": [
            {"step_id": "S001", "combined_formal_translation": "x is one",
             "missing_clauses": issue, "weakened_clauses": [],
             "unbound_objects": [], "wrong_relations": []},
            {"step_id": "S002", "combined_formal_translation": "x equals one",
             "missing_clauses": [], "weakened_clauses": [],
             "unbound_objects": [], "wrong_relations": []},
        ],
        "root": {"translation": "sourceValue equals one",
                 "target_object_preserved": True, "answer_grounded": True,
                 "reasons": []},
        "unreachable_steps": [], "dependency_issues": [],
        "obligation_results": list(obligations),
    })


def _response(content: str, *, finish_reason: str = "stop"):
    return SimpleNamespace(
        id="request-1",
        choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(content=content, reasoning_content=None, model_extra={}),
        )],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=30, total_tokens=130),
    )


class SemanticAuditV2Test(unittest.TestCase):
    def test_formal_view_excludes_attributes_comments_metadata_and_proofs(self) -> None:
        _manifest, blueprint = _fixture()
        view = build_formal_view(blueprint)
        rendered = formal_decompiler_messages(view)[1]["content"]
        self.assertIn("def sourceValue : Nat := 1", rendered)
        self.assertIn("theorem setup : sourceValue = 1", rendered)
        self.assertNotIn("blueprint", rendered)
        self.assertNotIn("metadata", rendered)
        self.assertNotIn("proof prose", rendered)
        self.assertNotIn("sorry_using", rendered)
        self.assertNotIn("hidden", rendered)

    def test_decompiler_requires_exact_ordered_node_inventory(self) -> None:
        _manifest, blueprint = _fixture()
        view = build_formal_view(blueprint)
        parsed = parse_formal_decompiler(_decompiler_json(), view=view)
        self.assertEqual([node.node_name for node in parsed], [
            "sourceValue", "setup", "root",
        ])
        value = json.loads(_decompiler_json())
        value["nodes"].reverse()
        with self.assertRaises(SemanticAuditFormatError):
            parse_formal_decompiler(json.dumps(value), view=view)

    def test_comparator_requires_complete_step_and_obligation_inventory(self) -> None:
        manifest, blueprint = _fixture()
        view = build_formal_view(blueprint)
        decompiler = _decompiler_result(view)
        obligations = [{"obligation_id": "semantic:x", "category": "missing",
                        "step_id": "S001", "node_names": ["setup"],
                        "requirement": "retain x", "reason": "missing", "status": "open"}]
        raw = _comparator_json(obligations=[{
            "obligation_id": "semantic:x", "resolved": True, "reason": "restored",
        }])
        parsed = parse_strict_comparator(
            raw, manifest=manifest, view=view, decompiler=decompiler,
            open_obligations=obligations,
        )
        self.assertTrue(parsed[-1])
        value = json.loads(raw)
        value["steps"].pop()
        with self.assertRaises(SemanticAuditFormatError):
            parse_strict_comparator(
                json.dumps(value), manifest=manifest, view=view, decompiler=decompiler,
                open_obligations=obligations,
            )

    def test_vacuous_decompiler_node_cannot_pass(self) -> None:
        manifest, blueprint = _fixture()
        view = build_formal_view(blueprint)
        decompiler = _decompiler_result(view, vacuous="setup")
        parsed = parse_strict_comparator(
            _comparator_json(), manifest=manifest, view=view,
            decompiler=decompiler, open_obligations=[],
        )
        self.assertFalse(parsed[-1])

    def test_format_retry_aggregates_usage_and_disables_thinking(self) -> None:
        _manifest, blueprint = _fixture()
        view = build_formal_view(blueprint)
        with patch("semantic_audit.chat_completion_with_retry", side_effect=[
            _response("not json"), _response(_decompiler_json()),
        ]) as request:
            result = run_formal_decompiler(
                object(), "model", view=view, max_tokens=4096, max_attempts=2,
            )
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(result.total_tokens, 260)
        self.assertEqual(request.call_count, 2)
        self.assertNotIn("response_format", request.call_args.kwargs)
        self.assertFalse(
            request.call_args.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        )

    def test_obligations_persist_then_resolve(self) -> None:
        manifest, blueprint = _fixture()
        view = build_formal_view(blueprint)
        decompiler = _decompiler_result(view)
        parsed = parse_strict_comparator(
            _comparator_json(missing=True), manifest=manifest, view=view,
            decompiler=decompiler, open_obligations=[],
        )
        first = StrictComparatorResult(*parsed[:-1], parsed[-1], "", "", "stop", "r", 0, 0, 0)
        ledger = {}
        open_items = _update_semantic_obligations(
            ledger, view=view, decompiler=decompiler, comparator=first,
            semantic_manifest=manifest, round_index=1, tracer=None, thm_name="x",
        )
        self.assertEqual(len(open_items), 1)
        oid = open_items[0]["obligation_id"]
        changed = json.loads(_comparator_json(missing=True, obligations=[{
            "obligation_id": oid, "resolved": False, "reason": "still absent",
        }]))
        changed["steps"][0]["missing_clauses"][0]["clause"] = "the source object x is bound"
        parsed_changed = parse_strict_comparator(
            json.dumps(changed), manifest=manifest, view=view, decompiler=decompiler,
            open_obligations=open_items,
        )
        still_open = _update_semantic_obligations(
            ledger, view=view, decompiler=decompiler,
            comparator=StrictComparatorResult(
                *parsed_changed[:-1], parsed_changed[-1], "", "", "stop", "r", 0, 0, 0,
            ),
            semantic_manifest=manifest, round_index=2, tracer=None, thm_name="x",
        )
        self.assertEqual([item["obligation_id"] for item in still_open], [oid])
        parsed = parse_strict_comparator(
            _comparator_json(obligations=[{
                "obligation_id": oid, "resolved": True, "reason": "fixed",
            }]), manifest=manifest, view=view, decompiler=decompiler,
            open_obligations=open_items,
        )
        second = StrictComparatorResult(*parsed[:-1], parsed[-1], "", "", "stop", "r", 0, 0, 0)
        self.assertEqual(_update_semantic_obligations(
            ledger, view=view, decompiler=decompiler, comparator=second,
            semantic_manifest=manifest, round_index=3, tracer=None, thm_name="x",
        ), ())

    def test_comparator_prompt_says_wrong_cot_is_not_a_truth_failure(self) -> None:
        manifest, blueprint = _fixture()
        view = build_formal_view(blueprint)
        messages = strict_comparator_messages(
            "problem", "1", manifest, view, _decompiler_result(view), [],
        )
        self.assertIn("mathematically\nwrong", messages[0]["content"])
        self.assertNotIn("metadata must not reach", messages[1]["content"])

    def test_prompts_preserve_repeated_final_claim_and_close_repaired_obligation(self) -> None:
        manifest, blueprint = _fixture()
        view = build_formal_view(blueprint)
        decompiler_prompt = formal_decompiler_messages(view)[0]["content"]
        self.assertIn("redundancy is not semantic vacuity", decompiler_prompt)
        obligations = [{
            "obligation_id": "semantic:vacuousNode:x",
            "category": "vacuousNode",
            "step_id": "S002",
            "node_names": ["root"],
            "requirement": "replace vacuous node",
            "reason": "old node was vacuous",
            "status": "open",
        }]
        comparator_prompt = strict_comparator_messages(
            "problem", "1", manifest, view, _decompiler_result(view), obligations,
        )[0]["content"]
        self.assertIn("persistent repair question, not an immutable verdict", comparator_prompt)
        self.assertIn("return\n`resolved:true`", comparator_prompt)
        self.assertIn("at most 25\nwords", comparator_prompt)


class Phase1ASkeletonTest(unittest.TestCase):
    def test_plain_local_definition_and_non_blueprint_dependency_are_rejected(self) -> None:
        blueprint = _parse_blueprint('''import Mathlib
import Architect
def PendingBlueprintClaim (_nodeId : String) : Prop := True
def hidden : Nat := 1
@[blueprint (title := "COT_STEP:S001") (statement := /-- pending -/)
  (proof := /-- use hidden -/)]
lemma setup : PendingBlueprintClaim "setup" := by sorry_using [hidden]
@[blueprint (title := "COT_STEP:S002") (statement := /-- result -/)
  (proof := /-- use setup -/)]
theorem root : PendingBlueprintClaim "root" := by sorry_using [setup]
''', "root")
        errors = _phase1a_contract_errors(blueprint)
        self.assertTrue(any(error.startswith("unannotatedLocalDeclaration:") for error in errors))
        self.assertTrue(any(error.startswith("nonBlueprintDependency:") for error in errors))

    def test_definition_cannot_use_pending_claim(self) -> None:
        blueprint = _parse_blueprint('''import Mathlib
import Architect
def PendingBlueprintClaim (_nodeId : String) : Prop := True
@[blueprint (title := "COT_STEP:S001") (statement := /-- setup -/)]
def setup : Prop := PendingBlueprintClaim "setup"
@[blueprint (title := "COT_STEP:S002") (statement := /-- result -/)
  (proof := /-- use setup -/)]
theorem root : PendingBlueprintClaim "root" := by sorry_using [setup]
''', "root")
        self.assertTrue(any(
            error.startswith("pendingDefinition:")
            for error in _phase1a_contract_errors(blueprint)
        ))

    def test_concrete_node_with_pending_ancestor_gets_helper_in_preflight(self) -> None:
        blueprint = _parse_blueprint('''import Mathlib
import Architect
def PendingBlueprintClaim (_nodeId : String) : Prop := True
@[blueprint (title := "COT_STEP:S001") (statement := /-- setup -/)
  (proof := /-- pending -/)]
lemma setup : PendingBlueprintClaim "setup" := by sorry_using []
@[blueprint (title := "COT_STEP:S002") (statement := /-- result -/)
  (proof := /-- use setup -/)]
theorem root : (1 : Nat) = 1 := by sorry_using [setup]
''', "root")
        preflight = _phase2_preflight_file(blueprint, blueprint.node_by_name("root"))
        self.assertIn("def PendingBlueprintClaim", preflight)
        self.assertLess(preflight.index("def PendingBlueprintClaim"), preflight.index("lemma setup"))


class SearchThenEditTest(unittest.TestCase):
    @staticmethod
    def _call(name: str, arguments: dict, call_id: str):
        function = SimpleNamespace(name=name, arguments=json.dumps(arguments))
        return SimpleNamespace(
            id=call_id,
            function=function,
            model_dump=lambda: {
                "id": call_id, "type": "function",
                "function": {"name": name, "arguments": function.arguments},
            },
        )

    @staticmethod
    def _model_response(calls):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=calls),
        )])

    def test_search_then_edit_uses_same_round_and_defers_mixed_edit(self) -> None:
        search = self._call("mathlib_search", {
            "query": "natural addition identity", "target_node_names": ["setup"], "k": 5,
        }, "search-1")
        mixed_edit = self._call("editBlueprintNode", {
            "action": "replace", "node_name": "setup", "expected_node_hash": "h",
            "replacement": "x", "reason": "r",
        }, "edit-deferred")
        final_edit = self._call("editBlueprintNode", {
            "action": "replace", "node_name": "setup", "expected_node_hash": "h",
            "replacement": "x", "reason": "r",
        }, "edit-2")
        responses = [self._model_response([search, mixed_edit]), self._model_response([final_edit])]
        retrieval = SimpleNamespace(search=lambda *_args: [
            LemmaResult("Nat.add_zero", "n + 0 = n", None),
        ])
        events = []
        with patch("blueprint._call_phase1b_model", side_effect=responses) as model_call:
            response = _phase1b_search_or_edit_response(
                object(), "model", [{"role": "user", "content": "repair"}],
                retrieval=retrieval, search_cache={}, search_limit=3, edit_limit=32,
                round_index=9, max_rounds=16, tracer=SimpleNamespace(emit=events.append),
                thm_name="sample",
            )
        self.assertIs(response, responses[1])
        self.assertEqual(model_call.call_count, 2)
        self.assertIn("Mode: EDIT ONLY", model_call.call_args.args[2][-1]["content"])
        deferred = [event for event in events if event.kind == "tool_result"
                    and event.call_id == "edit-deferred"]
        self.assertEqual(deferred[0].result, "editDeferredUntilAfterSearch")

    def test_search_limit_and_query_cache_are_enforced(self) -> None:
        calls = [self._call("mathlib_search", {
            "query": "same query", "target_node_names": ["setup"], "k": 5,
        }, f"search-{index}") for index in range(4)]
        counter = {"calls": 0}

        def search(*_args):
            counter["calls"] += 1
            return [LemmaResult("Nat.add_zero", "n + 0 = n", "")]

        with patch("blueprint._call_phase1b_model", side_effect=[
            self._model_response(calls), self._model_response([]),
        ]):
            _phase1b_search_or_edit_response(
                object(), "model", [{"role": "user", "content": "repair"}],
                retrieval=SimpleNamespace(search=search), search_cache={}, search_limit=3,
                edit_limit=32, round_index=1, max_rounds=16, tracer=None,
                thm_name="sample",
            )
        self.assertEqual(counter["calls"], 1)


if __name__ == "__main__":
    unittest.main()
