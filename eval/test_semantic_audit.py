from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from semantic_audit import (  # noqa: E402
    SemanticAuditFormatError,
    build_semantic_audit_messages,
    parse_semantic_audit,
    run_semantic_audit,
    semantic_audit_formal_view,
)


class RecordingTracer:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


class FakeCompletions:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def response(
    content: str,
    *,
    reasoning: str = "private audit reasoning",
    finish_reason: str = "stop",
    prompt_tokens: int = 101,
    completion_tokens: int = 17,
):
    return SimpleNamespace(
        id="audit-response",
        request_id="request-123",
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=content,
                reasoning_content=reasoning,
                model_extra={},
            ),
            finish_reason=finish_reason,
        )],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def client_for(response_value):
    completions = FakeCompletions(response_value)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


class SemanticAuditParserTest(unittest.TestCase):
    def test_formal_view_keeps_definitions_and_signatures_but_removes_proof_noise(self) -> None:
        source = """import Mathlib
-- prose outside the declaration
@[blueprint (title := "COT_STEP:S001") (statement := /-- informal claim -/)]
def N : ℕ := 27 ^ 3
@[blueprint (title := "COT_STEP:S002") (statement := /-- claim -/) (proof := /-- argument -/)]
lemma bridge : N = 27 ^ 3 := by sorry_using [N, helper]
"""
        view = semantic_audit_formal_view(source)

        self.assertIn('title := "COT_STEP:S001"', view)
        self.assertIn("def N : ℕ := 27 ^ 3", view)
        self.assertIn("lemma bridge : N = 27 ^ 3 := by sorry", view)
        self.assertNotIn("informal claim", view)
        self.assertNotIn("argument", view)
        self.assertNotIn("helper", view)
        self.assertNotIn("prose outside", view)

    def test_formal_view_removes_string_and_doc_comment_prose_fields(self) -> None:
        source = r'''@[blueprint
  (title := "COT_STEP:S001")
  (statement := "natural language with an escaped \"quote\"")
  (proof := /-- another prose explanation -/)]
lemma source_claim : x = 3 := by sorry_using [helper]
'''
        view = semantic_audit_formal_view(source)

        self.assertIn('title := "COT_STEP:S001"', view)
        self.assertIn("lemma source_claim : x = 3 := by sorry", view)
        self.assertNotIn("natural language", view)
        self.assertNotIn("another prose", view)
        self.assertNotIn("helper", view)

    def test_parses_unique_first_line_pass(self) -> None:
        parsed = parse_semantic_audit("  [[SEMANTIC_AUDIT=PASS]]\n")
        self.assertTrue(parsed.passed)
        self.assertEqual(parsed.flag, "PASS")
        self.assertEqual(parsed.diagnostics, "")

    def test_parses_fail_with_step_diagnostics(self) -> None:
        parsed = parse_semantic_audit(
            "[[SEMANTIC_AUDIT=FAIL]]\n"
            "S004: node reverses the source inequality.\n"
            "S007: root hard-codes the answer."
        )
        self.assertFalse(parsed.passed)
        self.assertEqual(parsed.flag, "FAIL")
        self.assertIn("S004", parsed.diagnostics)
        self.assertIn("S007", parsed.diagnostics)

    def test_rejects_missing_duplicate_conflicting_and_nonfirst_flags(self) -> None:
        invalid = {
            "missing": "S001: no flag",
            "duplicate": "[[SEMANTIC_AUDIT=PASS]]\n[[SEMANTIC_AUDIT=PASS]]",
            "conflicting": "[[SEMANTIC_AUDIT=PASS]]\n[[SEMANTIC_AUDIT=FAIL]]",
            "not the first line": "S001: mismatch\n[[SEMANTIC_AUDIT=FAIL]]",
        }
        for reason, content in invalid.items():
            with self.subTest(reason=reason):
                with self.assertRaises(SemanticAuditFormatError) as caught:
                    parse_semantic_audit(content)
                self.assertIn(reason, caught.exception.reason)
                self.assertEqual(caught.exception.raw_content, content)

    def test_claim_inventory_is_complete_ordered_and_consistent_with_flag(self) -> None:
        claim_ids = ("S001.C001", "S002.C001")
        passed = parse_semantic_audit(
            "[[SEMANTIC_AUDIT=PASS]]\n"
            "[[CLAIMS=S001.C001:OK,S002.C001:OK]]",
            expected_claim_ids=claim_ids,
        )
        self.assertTrue(passed.passed)
        self.assertEqual(
            passed.claim_statuses,
            (("S001.C001", "OK"), ("S002.C001", "OK")),
        )

        invalid = (
            "[[SEMANTIC_AUDIT=PASS]]\n[[CLAIMS=S001.C001:OK]]",
            "[[SEMANTIC_AUDIT=PASS]]\n"
            "[[CLAIMS=S001.C001:OK,S002.C001:MISSING]]",
            "[[SEMANTIC_AUDIT=FAIL]]\n"
            "[[CLAIMS=S001.C001:OK,S002.C001:OK]]",
        )
        for content in invalid:
            with self.subTest(content=content):
                with self.assertRaises(SemanticAuditFormatError):
                    parse_semantic_audit(content, expected_claim_ids=claim_ids)

        failed = parse_semantic_audit(
            "[[SEMANTIC_AUDIT=FAIL]]\n"
            "[[CLAIMS=S001.C001:OK,S002.C001:MISMATCH]]\n"
            "S002.C001: the Lean signature adds a converse.",
            expected_claim_ids=claim_ids,
        )
        self.assertFalse(failed.passed)
        self.assertEqual(failed.diagnostics, "S002.C001: the Lean signature adds a converse.")


class SemanticAuditRequestTest(unittest.TestCase):
    def test_uses_openai_completion_id_as_request_id_fallback(self) -> None:
        fake = response("[[SEMANTIC_AUDIT=PASS]]")
        fake.request_id = None
        fake.id = "chatcmpl-live-123"
        with patch("semantic_audit.chat_completion_with_retry", return_value=fake):
            result = run_semantic_audit(
                "model", "[COT_STEP S001]x[/COT_STEP S001]", "theorem t : True",
                client=object(),
            )
        self.assertEqual(result.request_id, "chatcmpl-live-123")

    def test_prompt_batches_all_inputs_and_modes_are_validated(self) -> None:
        messages = build_semantic_audit_messages(
            "[COT_STEP S001] source claim",
            "theorem root : True",
            mode="full",
            informal_statement="For the original integer n, find n + 1.",
            claimed_answer="7",
        )
        prompt = "\n".join(message["content"] for message in messages)
        normalized_prompt = " ".join(prompt.split())
        self.assertIn("Audit mode: full", prompt)
        self.assertIn("Original informal problem statement", prompt)
        self.assertIn("For the original integer n, find n + 1.", prompt)
        self.assertIn("Claimed answer from the original COT", prompt)
        self.assertIn("\n7\n", prompt)
        self.assertIn("[COT_STEP S001] source claim", prompt)
        self.assertIn("theorem root : True", prompt)
        self.assertIn(
            "faithfully translated mathematically wrong COT must PASS",
            normalized_prompt,
        )
        self.assertIn("root asks the original problem's question", prompt)
        self.assertIn("do not repair or re-grade it", prompt)
        self.assertIn("FIRST LINE", prompt)
        self.assertIn("do not output chain-of-thought", normalized_prompt.lower())
        with self.assertRaisesRegex(ValueError, "risk.*full"):
            build_semantic_audit_messages("cot", "lean", mode="invalid")  # type: ignore[arg-type]

    def test_prompt_calibrates_root_dependencies_and_specialization(self) -> None:
        messages = build_semantic_audit_messages(
            "[COT_STEP S001] Apply the claim at x = 2.\n"
            "[COT_STEP S002] Therefore the requested value is 4.",
            "theorem root : requestedValue = 4 := by sorry_using [step_s002]",
            mode="full",
            informal_statement="Find the requested value.",
            claimed_answer="4",
        )
        prompt = " ".join(
            message["content"] for message in messages
        )
        normalized_prompt = " ".join(prompt.split())

        self.assertIn("proposition-coverage auditor, not a proof judge", normalized_prompt)
        self.assertIn("deterministic checker has already verified step IDs", normalized_prompt)
        self.assertIn("Never judge whether a node is used", normalized_prompt)
        self.assertIn("specialization used by the COT", normalized_prompt)
        self.assertIn("at most two diagnostic lines", normalized_prompt)
        self.assertIn("at most 45 words", normalized_prompt)
        self.assertIn("Do not duplicate a root cause", normalized_prompt)

    def test_prompt_treats_explicit_gaps_and_source_assumptions_as_faithful(self) -> None:
        messages = build_semantic_audit_messages(
            "[COT_STEP S001] Assume O lies on the diagonal.\n"
            "[COT_STEP S002] The count is N = 27^3.",
            "lemma count_bridge : N = 27^3 := by sorry_using [partial_count]",
            mode="full",
            informal_statement="Count the valid triples.",
            claimed_answer="683",
        )
        prompt = " ".join(message["content"] for message in messages)
        normalized_prompt = " ".join(prompt.split())

        self.assertIn("every displayed `by sorry` means", normalized_prompt)
        self.assertIn("N = 27^3", normalized_prompt)
        self.assertIn("no missing-justification lemma is required", normalized_prompt)
        self.assertIn("source `assume` may become a hypothesis", normalized_prompt)
        self.assertIn("object identity/binding", normalized_prompt)
        self.assertIn("Mathematical truth and provability are irrelevant", normalized_prompt)
        self.assertIn("Inventory every mathematical proposition", normalized_prompt)
        self.assertIn("whether the root logically follows", normalized_prompt)
        self.assertIn("restrictedCount = K", normalized_prompt)
        self.assertIn("formal-only Lean view", normalized_prompt)
        self.assertIn("total_valid_triples : N = 27^3", normalized_prompt)
        self.assertIn("N = restrictedCount", normalized_prompt)
        self.assertIn("r_i := 9", normalized_prompt)

    def test_uses_supplied_client_env_budget_and_emits_usage_and_response(self) -> None:
        fake_client, completions = client_for(response("[[SEMANTIC_AUDIT=PASS]]"))
        tracer = RecordingTracer()
        with patch.dict(os.environ, {"GOEDEL_SEMANTIC_AUDIT_MAX_TOKENS": "321"}):
            result = run_semantic_audit(
                "audit-model",
                "[COT_STEP S001] x = 1",
                "theorem root : x = 1",
                mode="risk",
                informal_statement="Given the original real number x, find x.",
                claimed_answer="x = 1",
                client=fake_client,
                tracer=tracer,
                thm_name="sample",
                phase="audit-phase",
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.request_id, "request-123")
        self.assertEqual(result.reasoning_content, "private audit reasoning")
        self.assertEqual((result.prompt_tokens, result.completion_tokens, result.total_tokens), (101, 17, 118))
        self.assertEqual(len(completions.calls), 1)
        request = completions.calls[0]
        self.assertEqual(request["max_completion_tokens"], 321)
        self.assertEqual(request["temperature"], 0)
        self.assertNotIn("response_format", request)
        request_prompt = "\n".join(
            message["content"] for message in request["messages"]
        )
        self.assertIn("Given the original real number x, find x.", request_prompt)
        self.assertIn("x = 1", request_prompt)
        self.assertEqual(
            request["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        kinds = [event.kind for event in tracer.events]
        self.assertIn("llm_request_start", kinds)
        self.assertIn("llm_request_end", kinds)
        self.assertIn("llm_usage", kinds)
        self.assertIn("llm_response", kinds)
        response_event = next(event for event in tracer.events if event.kind == "llm_response")
        self.assertEqual(response_event.result, "[[SEMANTIC_AUDIT=PASS]]")

    def test_creates_client_when_none_is_supplied(self) -> None:
        fake_client, completions = client_for(
            response("[[SEMANTIC_AUDIT=FAIL]]\nS003: conclusion changed.")
        )
        with patch("semantic_audit.make_client", return_value=fake_client) as make:
            result = run_semantic_audit(
                "audit-model", "numbered cot", "blueprint", mode="full", max_tokens=55,
            )

        make.assert_called_once_with("audit-model")
        self.assertFalse(result.passed)
        self.assertEqual(result.flag, "FAIL")
        self.assertIn("S003", result.diagnostics)
        self.assertEqual(completions.calls[0]["max_completion_tokens"], 55)

    def test_length_response_with_opening_flag_is_not_a_decision_but_is_recorded(self) -> None:
        raw = (
            "[[SEMANTIC_AUDIT=FAIL]]\n"
            "S002: source says x < y but the node states x <= y."
        )
        fake_client, _completions = client_for(
            response(raw, finish_reason="length", completion_tokens=1024)
        )
        tracer = RecordingTracer()

        with self.assertRaisesRegex(SemanticAuditFormatError, "truncated") as caught:
            run_semantic_audit(
                "audit-model", "numbered cot", "blueprint",
                client=fake_client, tracer=tracer,
            )

        self.assertEqual(caught.exception.raw_content, raw)
        self.assertEqual(caught.exception.markers, ("FAIL",))
        response_event = next(
            event for event in tracer.events if event.kind == "llm_response"
        )
        self.assertTrue(response_event.args["truncated"])

    def test_default_budget_is_1024(self) -> None:
        fake_client, completions = client_for(response("[[SEMANTIC_AUDIT=PASS]]"))
        with patch.dict(os.environ, {}, clear=True):
            run_semantic_audit(
                "audit-model", "numbered cot", "blueprint", client=fake_client,
            )
        self.assertEqual(completions.calls[0]["max_completion_tokens"], 1024)

    def test_format_error_preserves_response_trace(self) -> None:
        fake_client, _completions = client_for(response("No marker was emitted."))
        tracer = RecordingTracer()

        with self.assertRaises(SemanticAuditFormatError):
            run_semantic_audit(
                "audit-model", "numbered cot", "blueprint",
                client=fake_client, tracer=tracer,
            )

        response_events = [event for event in tracer.events if event.kind == "llm_response"]
        self.assertEqual(len(response_events), 1)
        self.assertEqual(response_events[0].result, "No marker was emitted.")


if __name__ == "__main__":
    unittest.main()
