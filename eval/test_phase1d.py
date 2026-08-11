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
from phase1d import (  # noqa: E402
    Phase1DRound,
    _accepted_validation_details,
    _d_contract_errors,
    _submitted_code,
    phase1d_request_budget,
    phase1d_round_classification,
)


class _Tokenizer:
    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt):
        self.payload = (messages, tools, tokenize, add_generation_prompt)
        return {"input_ids": list(range(123)), "attention_mask": [1] * 123}


class Phase1DFullRegenerationTest(unittest.TestCase):
    def test_accepted_terminal_details_are_json_serializable(self) -> None:
        details = {"semanticAudit": {"strictComparator": {"passed": True}}}
        current_round = Phase1DRound(
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
        self.assertNotIn("phase1DRounds", details)
        self.assertEqual(terminal["classification"], "strictAccepted")

    def test_budget_counts_serialized_messages_and_tool_schema(self) -> None:
        tokenizer = _Tokenizer()
        with patch("phase1d._load_phase1_tokenizer", return_value=tokenizer):
            input_tokens, output_tokens = phase1d_request_budget(
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
        codes = {item["code"] for item in _d_contract_errors(blueprint, "root")}
        self.assertIn("forbiddenPendingClaim", codes)

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

    def test_warning_only_waits_until_last_round(self) -> None:
        self.assertIsNone(phase1d_round_classification(
            round_index=7, max_turns=8, deterministic_error_count=0,
            semantic_error_count=0, warning_count=1,
        ))
        self.assertEqual(phase1d_round_classification(
            round_index=8, max_turns=8, deterministic_error_count=0,
            semantic_error_count=0, warning_count=1,
        ), "acceptedWithWarnings")

    def test_error_precedence_and_strict_early_exit(self) -> None:
        self.assertEqual(phase1d_round_classification(
            round_index=1, max_turns=8, deterministic_error_count=0,
            semantic_error_count=0, warning_count=0,
        ), "strictAccepted")
        self.assertEqual(phase1d_round_classification(
            round_index=8, max_turns=8, deterministic_error_count=1,
            semantic_error_count=2, warning_count=0,
        ), "structuralRejected")
        self.assertEqual(phase1d_round_classification(
            round_index=8, max_turns=8, deterministic_error_count=0,
            semantic_error_count=2, warning_count=0,
        ), "semanticRejected")


if __name__ == "__main__":
    unittest.main()
