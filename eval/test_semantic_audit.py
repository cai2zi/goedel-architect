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
    COMPARATOR_SYSTEM_PROMPT,
    FormalDecompilerResult,
    SemanticAuditFormatError,
    build_formal_view,
    parse_formal_decompiler,
    parse_strict_comparator,
    parse_whole_cot_comparator,
    run_formal_decompiler,
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


def _manifest():
    return SimpleNamespace(steps=(
        SimpleNamespace(step_id="S001", source_text="Set the source value to 6."),
        SimpleNamespace(step_id="S002", source_text="Therefore the answer is 6."),
    ))


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


def _comparator_payload(*, missing: bool = False, root_ok: bool = True):
    issue = ({
        "clause": "sourceValue must be bound to the source object",
        "node_names": ["root"],
        "reason": "The source object is replaced.",
    } if missing else None)
    return {
        "steps": [
            {
                "step_id": "S001",
                "combined_formal_translation": "Defines sourceValue as six.",
                "missing_clauses": [],
                "weakened_clauses": [],
                "unbound_objects": [],
                "wrong_relations": [],
            },
            {
                "step_id": "S002",
                "combined_formal_translation": "Concludes sourceValue equals six.",
                "missing_clauses": [issue] if issue else [],
                "weakened_clauses": [],
                "unbound_objects": [],
                "wrong_relations": [],
            },
        ],
        "root": {
            "translation": "sourceValue equals six.",
            "target_object_preserved": root_ok,
            "answer_grounded": root_ok,
            "reasons": [] if root_ok else ["The root replaces the target object."],
        },
        "unreachable_steps": [],
        "dependency_issues": [],
        "obligation_results": [],
    }


class SemanticAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.view = build_formal_view(_parse_blueprint(LEAN, "root"))
        self.manifest = _manifest()

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

    def test_comparator_pass_is_computed_from_all_defects(self) -> None:
        decompiler = _decompiler(self.view)
        parsed = parse_strict_comparator(
            json.dumps(_comparator_payload()), manifest=self.manifest,
            view=self.view, decompiler=decompiler, open_obligations=(),
        )
        self.assertTrue(parsed[-1])

        parsed = parse_strict_comparator(
            json.dumps(_comparator_payload(missing=True)), manifest=self.manifest,
            view=self.view, decompiler=decompiler, open_obligations=(),
        )
        self.assertFalse(parsed[-1])

        parsed = parse_strict_comparator(
            json.dumps(_comparator_payload()), manifest=self.manifest,
            view=self.view, decompiler=_decompiler(self.view, vacuous=True),
            open_obligations=(),
        )
        self.assertFalse(parsed[-1])

    def test_comparator_prompt_demands_object_and_direction_fidelity(self) -> None:
        self.assertIn("Give no credit to ex-falso", COMPARATOR_SYSTEM_PROMPT)
        self.assertIn("Track object identity across Steps", COMPARATOR_SYSTEM_PROMPT)
        self.assertIn("root must mention the shared target object", COMPARATOR_SYSTEM_PROMPT)

    def test_whole_cot_formal_view_and_comparator_have_no_step_inventory(self) -> None:
        view = build_formal_view(_parse_blueprint(LEAN, "root"), include_step_ids=False)
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
        view = build_formal_view(_parse_blueprint(LEAN, "root"), include_step_ids=False)
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


if __name__ == "__main__":
    unittest.main()
