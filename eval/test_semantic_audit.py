from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blueprint import _parse_blueprint  # noqa: E402
from semantic_audit import (  # noqa: E402
    FormalDecompilerResult,
    JOINT_SEMANTIC_EFFECT_ALIASES,
    JOINT_WHOLE_COT_SYSTEM_PROMPT,
    JOINT_WHOLE_COT_PROMPT_VERSION,
    SemanticAuditFormatError,
    WHOLE_COT_PROMPT_VERSION,
    build_formal_view,
    joint_whole_cot_audit_messages,
    parse_formal_decompiler,
    parse_joint_whole_cot_audit,
    parse_whole_cot_comparator,
    _response_parts,
    run_formal_decompiler,
    run_joint_whole_cot_audit,
    semantic_audit_cache_key,
    whole_cot_comparator_messages,
)


LEAN = r'''
import Mathlib
import Architect

@[blueprint (title := "COT_STEP:S001")
  (statement := /-- This comment must not be semantic evidence. -/)
  (proof := /-- Nor may this proof description count. -/)]
def sourceValue : Nat := 6

@[blueprint (title := "COT_STEP:S002")
  (statement := /-- Derive the answer. -/)
  (proof := /-- Uses sourceValue. -/)]
theorem root : sourceValue = 6 := by
  sorry_using [sourceValue]
'''


def _decompiler(view, *, vacuous: bool = False):
    payload = {
        "nodes": [
            {
                "node_name": "sourceValue",
                "kind": "definition",
                "translation": "Defines sourceValue as six.",
                "semantic_effect": "vacuous" if vacuous else "objectDefinition",
                "introduced_objects": ["sourceValue"],
                "referenced_objects": [],
            },
            {
                "node_name": "root",
                "kind": "theorem",
                "translation": "Concludes that sourceValue equals six.",
                "semantic_effect": "proposition",
                "introduced_objects": [],
                "referenced_objects": ["sourceValue"],
            },
        ]
    }
    raw = json.dumps(payload)
    return FormalDecompilerResult(
        parse_formal_decompiler(raw, view=view), raw, "", "stop", "request", 1, 1, 2,
    )


def _whole_cot_payload(*, missing: bool = False, root_ok: bool = True):
    issue = ({
        "clause": "bind original object", "node_names": ["root"],
        "reason": "target object replaced",
    } if missing else None)
    return {
        "cot": {
            "combined_formal_translation": "Defines and uses sourceValue.",
            "missing_clauses": [issue] if issue else [],
            "weakened_clauses": [], "unbound_objects": [],
            "wrong_relations": [], "added_clauses": [],
        },
        "root": {
            "translation": "sourceValue equals six.",
            "target_object_preserved": root_ok,
            "answer_grounded": root_ok,
            "reasons": [] if root_ok else ["target object replaced"],
        },
        "unreachable_nodes": [], "dependency_issues": [],
    }


def _joint_payload(view, *, missing: bool = False, root_ok: bool = True):
    return {
        "formal_decompiler": json.loads(_decompiler(view).raw_content),
        "whole_cot_comparator": _whole_cot_payload(
            missing=missing, root_ok=root_ok,
        ),
    }


class SemanticAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.view = build_formal_view(_parse_blueprint(LEAN, "root"))

    def test_formal_view_strips_metadata_comments_and_proof_body(self) -> None:
        declarations = {node.node_name: node.declaration for node in self.view.nodes}
        self.assertEqual(declarations["sourceValue"], "def sourceValue : Nat := 6")
        self.assertEqual(declarations["root"], "theorem root : sourceValue = 6")
        self.assertNotIn("blueprint", "\n".join(declarations.values()))
        self.assertNotIn("sorry_using", declarations["root"])
        self.assertEqual(self.view.root_closure, ("sourceValue", "root"))

    def test_decompiler_requires_exact_ordered_node_inventory(self) -> None:
        decompiler = _decompiler(self.view)
        self.assertEqual([node.node_name for node in decompiler.nodes], ["sourceValue", "root"])
        value = json.loads(decompiler.raw_content)
        value["nodes"].reverse()
        with self.assertRaises(SemanticAuditFormatError):
            parse_formal_decompiler(json.dumps(value), view=self.view)

    def test_separate_decompiler_keeps_strict_semantic_effect_enum(self) -> None:
        value = json.loads(_decompiler(self.view).raw_content)
        value["nodes"][1]["semantic_effect"] = "theoremStatement"
        with self.assertRaisesRegex(
            SemanticAuditFormatError, "invalid semantic_effect theoremStatement",
        ):
            parse_formal_decompiler(json.dumps(value), view=self.view)

    def test_whole_cot_formal_view_and_comparator_have_no_step_inventory(self) -> None:
        view = build_formal_view(_parse_blueprint(LEAN, "root"))
        decompiler = _decompiler(view)
        payload = {
            "cot": {
                "combined_formal_translation": "Defines and uses sourceValue.",
                "missing_clauses": [], "weakened_clauses": [],
                "unbound_objects": [], "wrong_relations": [], "added_clauses": [],
            },
            "root": {
                "translation": "sourceValue equals six.",
                "target_object_preserved": True, "answer_grounded": True,
                "reasons": [],
            },
            "unreachable_nodes": [], "dependency_issues": [],
        }
        parsed = parse_whole_cot_comparator(
            json.dumps(payload), view=view, decompiler=decompiler,
        )
        self.assertTrue(parsed[-1])
        serialized = json.dumps(view.to_dict())
        messages = json.dumps(whole_cot_comparator_messages(
            "problem", "raw cot", "6", view, decompiler,
        ))
        self.assertNotIn("step_id", serialized)
        self.assertNotIn("COT_STEP", messages)

    def test_whole_cot_comparator_rejects_missing_clause(self) -> None:
        view = build_formal_view(_parse_blueprint(LEAN, "root"))
        decompiler = _decompiler(view)
        issue = {"clause": "bind original object", "node_names": ["root"], "reason": "replaced"}
        payload = {
            "cot": {
                "combined_formal_translation": "Only the answer.",
                "missing_clauses": [issue], "weakened_clauses": [],
                "unbound_objects": [], "wrong_relations": [], "added_clauses": [],
            },
            "root": {"translation": "six", "target_object_preserved": False,
                     "answer_grounded": False, "reasons": ["object replaced"]},
            "unreachable_nodes": [], "dependency_issues": [],
        }
        self.assertFalse(parse_whole_cot_comparator(
            json.dumps(payload), view=view, decompiler=decompiler,
        )[-1])

    def test_semantic_audit_forwards_thinking_sampling(self) -> None:
        raw = _decompiler(self.view).raw_content
        response = SimpleNamespace(
            id="request",
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=raw, reasoning_content="reasoning"),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        with patch("semantic_audit.chat_completion_with_retry", return_value=response) as chat:
            run_formal_decompiler(
                object(), "model", view=self.view, max_tokens=16384, max_attempts=1,
                enable_thinking=True, temperature=0.6, top_p=0.95,
                top_k=20, min_p=0.0, presence_penalty=0.0,
                repetition_penalty=1.0,
            )
        kwargs = chat.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.6)
        self.assertEqual(kwargs["top_p"], 0.95)
        self.assertEqual(kwargs["presence_penalty"], 0.0)
        self.assertEqual(kwargs["max_completion_tokens"], 16384)
        self.assertEqual(kwargs["extra_body"], {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True},
        })

    def test_response_parts_accepts_vllm_reasoning_field(self) -> None:
        response = SimpleNamespace(
            id="request",
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="{}", reasoning="visible thinking"),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        content, reasoning, finish, request_id, usage = _response_parts(response)
        self.assertEqual(content, "{}")
        self.assertEqual(reasoning, "visible thinking")
        self.assertEqual(finish, "stop")
        self.assertEqual(request_id, "request")
        self.assertEqual(usage, (10, 20, 30))

    def test_joint_schema_requires_order_and_known_nodes(self) -> None:
        view = build_formal_view(_parse_blueprint(LEAN, "root"))
        payload = _joint_payload(view)
        parsed = parse_joint_whole_cot_audit(json.dumps(payload), view=view)
        self.assertTrue(parsed[-1])
        reversed_payload = {
            "whole_cot_comparator": payload["whole_cot_comparator"],
            "formal_decompiler": payload["formal_decompiler"],
        }
        with self.assertRaisesRegex(SemanticAuditFormatError, "must be formal_decompiler"):
            parse_joint_whole_cot_audit(json.dumps(reversed_payload), view=view)
        payload["whole_cot_comparator"]["dependency_issues"] = [
            {"node_name": "unknown", "reason": "bad reference"}
        ]
        with self.assertRaisesRegex(SemanticAuditFormatError, "unknown node"):
            parse_joint_whole_cot_audit(json.dumps(payload), view=view)

    def test_joint_normalizes_all_effect_aliases_observed_in_failed_runs(self) -> None:
        view = build_formal_view(_parse_blueprint(LEAN, "root"))
        proposition_aliases = {
            "theoremStatement", "propertyAssertion", "theoremAssertion",
            "lemmaStatement", "claimAssertion", "statementAssertion",
            "propositionStatement", "propertyStatement", "propertyClaim",
            "mathematicalAssertion", "logicalStatement", "logicalClaim",
            "logicalAssertion", "lemma", "deduction", "claim", "assumption",
            "assertsProperty", "theoremClaim", "states_property",
            "mathematicalStatement", "claimStatement", "asserts_property",
        }
        object_aliases = {"propertyDefinition", "definition", "defines_constant"}
        self.assertEqual(
            set(JOINT_SEMANTIC_EFFECT_ALIASES),
            {
                "".join(character for character in alias.lower() if character.isalnum())
                for alias in proposition_aliases | object_aliases
            },
        )
        for alias in sorted(proposition_aliases):
            with self.subTest(alias=alias):
                payload = _joint_payload(view)
                payload["formal_decompiler"]["nodes"][1]["semantic_effect"] = alias
                parsed = parse_joint_whole_cot_audit(json.dumps(payload), view=view)
                self.assertEqual(parsed[0][1].semantic_effect, "proposition")
        for alias in sorted(object_aliases):
            with self.subTest(alias=alias):
                payload = _joint_payload(view)
                payload["formal_decompiler"]["nodes"][0]["semantic_effect"] = alias
                parsed = parse_joint_whole_cot_audit(json.dumps(payload), view=view)
                self.assertEqual(parsed[0][0].semantic_effect, "objectDefinition")

    def test_joint_rejects_unknown_effect_alias(self) -> None:
        view = build_formal_view(_parse_blueprint(LEAN, "root"))
        payload = _joint_payload(view)
        payload["formal_decompiler"]["nodes"][1]["semantic_effect"] = "someNewLabel"
        with self.assertRaisesRegex(
            SemanticAuditFormatError, "invalid semantic_effect someNewLabel",
        ):
            parse_joint_whole_cot_audit(json.dumps(payload), view=view)

    def test_joint_alias_cannot_hide_obvious_true_shell(self) -> None:
        lean = LEAN.replace("def sourceValue : Nat := 6", "def sourceValue : Prop := True")
        view = build_formal_view(_parse_blueprint(lean, "root"))
        payload = _joint_payload(view)
        payload["formal_decompiler"]["nodes"][0]["semantic_effect"] = "propertyDefinition"
        parsed = parse_joint_whole_cot_audit(json.dumps(payload), view=view)
        self.assertEqual(parsed[0][0].semantic_effect, "vacuous")
        self.assertFalse(parsed[-1])

    def test_joint_prompt_repeats_closed_enum_and_compact_limits(self) -> None:
        self.assertIn("closed enum", JOINT_WHOLE_COT_SYSTEM_PROMPT)
        self.assertIn("`objectDefinition`, `proposition`, or `vacuous`", JOINT_WHOLE_COT_SYSTEM_PROMPT)
        self.assertIn("Never emit alternatives", JOINT_WHOLE_COT_SYSTEM_PROMPT)
        self.assertIn("under 40 words", JOINT_WHOLE_COT_SYSTEM_PROMPT)

    def test_joint_pass_is_recomputed_locally(self) -> None:
        view = build_formal_view(_parse_blueprint(LEAN, "root"))
        parsed = parse_joint_whole_cot_audit(
            json.dumps(_joint_payload(view, missing=True)), view=view,
        )
        self.assertFalse(parsed[-1])

    def test_joint_uses_one_request_and_forwards_sampling(self) -> None:
        view = build_formal_view(_parse_blueprint(LEAN, "root"))
        response = SimpleNamespace(
            id="joint-request",
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(_joint_payload(view)),
                    reasoning_content="translate then compare",
                ),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=22, total_tokens=33),
        )
        with patch("semantic_audit.chat_completion_with_retry", return_value=response) as chat:
            result = run_joint_whole_cot_audit(
                object(), "model", informal_statement="problem",
                informal_proof="cot", claimed_answer="6", view=view,
                max_tokens=32768, max_attempts=2, enable_thinking=True,
                temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                presence_penalty=0.0, repetition_penalty=1.0,
            )
        self.assertEqual(chat.call_count, 1)
        self.assertTrue(result.comparator.passed)
        self.assertEqual(result.total_tokens, 33)
        self.assertEqual(chat.call_args.kwargs["max_completion_tokens"], 32768)
        self.assertEqual(
            chat.call_args.kwargs["extra_body"]["chat_template_kwargs"],
            {"enable_thinking": True},
        )

    def test_joint_result_records_effect_normalization(self) -> None:
        view = build_formal_view(_parse_blueprint(LEAN, "root"))
        payload = _joint_payload(view)
        payload["formal_decompiler"]["nodes"][1]["semantic_effect"] = "theoremStatement"
        response = SimpleNamespace(
            id="joint-alias",
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload), reasoning_content=""),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        with patch("semantic_audit.chat_completion_with_retry", return_value=response):
            result = run_joint_whole_cot_audit(
                object(), "model", informal_statement="problem",
                informal_proof="cot", claimed_answer="6", view=view,
                max_tokens=32768, max_attempts=2,
            )
        self.assertEqual(result.semantic_effect_normalizations, ({
            "node_name": "root", "reported": "theoremStatement",
            "canonical": "proposition",
        },))

    def test_joint_schema_retry_rebuilds_complete_response(self) -> None:
        view = build_formal_view(_parse_blueprint(LEAN, "root"))
        def response(content: str, request_id: str):
            return SimpleNamespace(
                id=request_id,
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=content, reasoning_content=""),
                    finish_reason="stop",
                )],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        with patch(
            "semantic_audit.chat_completion_with_retry",
            side_effect=[response("{}", "bad"), response(json.dumps(_joint_payload(view)), "ok")],
        ) as chat:
            result = run_joint_whole_cot_audit(
                object(), "model", informal_statement="problem",
                informal_proof="cot", claimed_answer="6", view=view,
                max_tokens=1000, max_attempts=2,
            )
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(len(result.attempts), 2)
        retry_messages = chat.call_args_list[1].kwargs["messages"]
        self.assertEqual(retry_messages[0]["role"], "system")
        self.assertIn("again from the original input", retry_messages[-1]["content"])
        self.assertIn("formal_decompiler, whole_cot_comparator", retry_messages[-1]["content"])

    def test_joint_cache_key_isolated_by_mode_and_sampling(self) -> None:
        view = build_formal_view(_parse_blueprint(LEAN, "root"))
        messages = joint_whole_cot_audit_messages("p", "cot", "6", view)
        joint = semantic_audit_cache_key(
            "model", messages, version=JOINT_WHOLE_COT_PROMPT_VERSION,
            request_params={"temperature": 0.6},
        )
        separate = semantic_audit_cache_key(
            "model", messages, version=WHOLE_COT_PROMPT_VERSION,
            request_params={"temperature": 0.6},
        )
        cold = semantic_audit_cache_key(
            "model", messages, version=JOINT_WHOLE_COT_PROMPT_VERSION,
            request_params={"temperature": 0.0},
        )
        self.assertEqual(len({joint, separate, cold}), 3)


if __name__ == "__main__":
    unittest.main()
