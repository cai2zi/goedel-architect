from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

from blueprint import _parse_blueprint  # noqa: E402
from blueprint_generation import (  # noqa: E402
    BlueprintValidation,
    GenerationRound,
    _accepted_validation_details,
    _contract_errors,
    _messages,
    _submitted_code,
    _validate_round,
    generation_request_budget,
    generation_round_classification,
)
from blueprint import Phase2StandaloneReport  # noqa: E402
from kimina_lean_compiler import CompilerResult  # noqa: E402
from semantic_fidelity import SemanticIssue  # noqa: E402


class _Tokenizer:
    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt):
        self.payload = (messages, tools, tokenize, add_generation_prompt)
        return {"input_ids": list(range(123)), "attention_mask": [1] * 123}


class BlueprintGenerationTest(unittest.TestCase):
    MINIMAL = '''import Mathlib
import Architect
@[blueprint] def model : Nat := 1
@[blueprint] theorem root : model = 1 := by sorry_using [model]
'''

    def test_accepted_terminal_details_are_json_serializable(self) -> None:
        details = {"semanticAudit": {"strictComparator": {"passed": True}}}
        current_round = GenerationRound(
            1, "candidate", 100, 200, (), (), (), details,
        )
        terminal = _accepted_validation_details(
            details,
            [current_round],
            classification="strictAccepted",
            deterministic_errors=(),
            semantic_errors=(),
            warnings=(),
        )

        json.dumps(terminal)
        self.assertNotIn("generationRounds", details)
        self.assertEqual(terminal["classification"], "strictAccepted")

    def test_budget_counts_serialized_messages_and_tool_schema(self) -> None:
        tokenizer = _Tokenizer()
        with patch("blueprint_generation._load_phase1_tokenizer", return_value=tokenizer):
            input_tokens, output_tokens = generation_request_budget(
                [{"role": "user", "content": "x"}],
                tokenizer_path="unused", model_max_context=40960,
                safety_margin=512, tools=[{"type": "function"}],
            )
        self.assertEqual(input_tokens, 123)
        self.assertEqual(output_tokens, 40960 - 123 - 512)
        self.assertEqual(tokenizer.payload[1], [{"type": "function"}])

    def test_pending_helper_or_conclusion_is_a_deterministic_error(self) -> None:
        code = '''import Mathlib
import Architect
def PendingBlueprintClaim (_nodeId : String) : Prop := True
@[blueprint (title := "COT_STEP:S001")
  (statement := /-- final -/) (proof := /-- final -/)]
theorem root : PendingBlueprintClaim "root" := by sorry_using []
'''
        blueprint = _parse_blueprint(code, "root")
        codes = {item["code"] for item in _contract_errors(blueprint, "root")}
        self.assertIn("forbiddenPendingClaim", codes)

    def test_minimal_metadata_is_not_required(self) -> None:
        blueprint = _parse_blueprint(self.MINIMAL, "root")
        self.assertEqual(_contract_errors(blueprint, "root"), [])

    def test_static_shadow_errors_do_not_block_acceptance(self) -> None:
        comparator = SimpleNamespace(passed=True)
        validation = BlueprintValidation(
            lean_result=CompilerResult(True),
            canonical_lean_result=CompilerResult(True),
            semantic_issues=[SemanticIssue("vacuous", "shadow only")],
            structural_errors=[],
            standalone_report=Phase2StandaloneReport((), 2, 0, 1.0),
            strict_comparator_result=comparator,
        )
        self.assertTrue(validation.passed)

    def test_whole_file_failure_short_circuits_semantic_audit(self) -> None:
        compiler = SimpleNamespace(check_blueprint=lambda *_: CompilerResult(
            False, errors=["bad syntax"], failure_kind="lean"
        ))
        with patch("blueprint_generation._with_semantic_audit") as audit:
            _blueprint, validation, deterministic, semantic, _warnings = _validate_round(
                self.MINIMAL, target_name="root", compiler=compiler,
                informal_statement="problem", informal_proof="cot", claimed_answer="1",
                standalone_concurrency=1, client=object(), model="model",
                decompiler_max_tokens=16, comparator_max_tokens=16,
                semantic_format_attempts=2, semantic_audit_enable_thinking=True,
                semantic_audit_temperature=0.6, semantic_audit_top_p=0.95,
                semantic_audit_top_k=20, semantic_audit_min_p=0.0,
                semantic_audit_presence_penalty=0.0,
                semantic_audit_repetition_penalty=1.0, tracer=None,
                thm_name="id", round_index=1, standalone_cache={},
                decompiler_cache={}, comparator_cache={},
            )
        audit.assert_not_called()
        self.assertEqual(validation.mechanical_failure_stage, "whole_file_lean")
        self.assertTrue(deterministic)
        self.assertFalse(semantic)

    def test_submission_requires_one_full_lean_compile_call(self) -> None:
        call = SimpleNamespace(function=SimpleNamespace(
            name="lean_compile", arguments=json.dumps({"lean_code": "theorem root : True := by trivial"}),
        ))
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(tool_calls=[call]),
        )])
        code, errors = _submitted_code(response)
        self.assertFalse(errors)
        self.assertTrue(code.endswith("\n"))

    def test_warning_only_is_immediately_accepted(self) -> None:
        self.assertEqual(generation_round_classification(
            round_index=7, max_turns=8, deterministic_error_count=0,
            semantic_error_count=0, warning_count=1,
        ), "strictAccepted")
        self.assertEqual(generation_round_classification(
            round_index=8, max_turns=8, deterministic_error_count=0,
            semantic_error_count=0, warning_count=1,
        ), "strictAccepted")

    def test_error_precedence_and_strict_early_exit(self) -> None:
        self.assertEqual(generation_round_classification(
            round_index=1, max_turns=8, deterministic_error_count=0,
            semantic_error_count=0, warning_count=0,
        ), "strictAccepted")
        self.assertEqual(generation_round_classification(
            round_index=8, max_turns=8, deterministic_error_count=1,
            semantic_error_count=2, warning_count=0,
        ), "structuralRejected")
        self.assertEqual(generation_round_classification(
            round_index=8, max_turns=8, deterministic_error_count=0,
            semantic_error_count=2, warning_count=0,
        ), "semanticRejected")

    def test_whole_cot_prompt_uses_raw_cot_without_step_contract(self) -> None:
        messages = _messages(
            target_name="root", informal_statement="problem",
            informal_proof="raw complete cot", claimed_answer="6",
            previous_blueprint="", previous_feedback="",
            prompt_profile="whole_cot_minimal",
        )
        rendered = "\n".join(item["content"] for item in messages)
        self.assertIn("raw complete cot", rendered)
        self.assertNotIn("COT_STEP:Snnn", rendered)
        self.assertIn("metadata are optional", rendered)


if __name__ == "__main__":
    unittest.main()
