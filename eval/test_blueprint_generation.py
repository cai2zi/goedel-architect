from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

from blueprint import BlueprintGenerationError, _parse_blueprint  # noqa: E402
from blueprint_generation import (  # noqa: E402
    BlueprintValidation,
    GenerationRound,
    _accepted_validation_details,
    _contract_errors,
    _messages,
    _submitted_code,
    _validate_round,
    _with_semantic_audit,
    generate_blueprint,
    generation_request_budget,
    generation_round_classification,
)
from blueprint import Phase2StandaloneReport  # noqa: E402
from kimina_lean_compiler import CompilerResult  # noqa: E402
from semantic_fidelity import SemanticIssue  # noqa: E402
from semantic_audit import (  # noqa: E402
    FormalDecompilerResult,
    JointWholeCotAuditResult,
    SemanticAuditFormatError,
    WholeCotComparatorResult,
)


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

    def test_generation_prompt_treats_definitions_as_global_context(self) -> None:
        messages = _messages(
            target_name="root", informal_statement="problem",
            informal_proof="cot", claimed_answer="1",
            previous_blueprint="", previous_feedback="",
        )
        system = " ".join(messages[0]["content"].split())
        self.assertIn("global context shared by every proof node", system)
        self.assertIn("if a definition is already listed", system)
        self.assertIn("mere existence does not ground", system)

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

    def test_formal_blueprint_failure_uses_one_compile_and_short_circuits_audit(self) -> None:
        check_blueprint = Mock(return_value=CompilerResult(
            False, errors=["bad syntax"], failure_kind="lean",
        ))
        compiler = SimpleNamespace(check_blueprint=check_blueprint)
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
                semantic_audit_repetition_penalty=1.0,
                semantic_audit_mode="separate",
                node_naming="semantic",
                joint_semantic_audit_max_tokens=32768,
                tokenizer_path="unused", model_max_context=40960,
                context_safety_margin=512, tracer=None,
                thm_name="id", round_index=1, standalone_cache={},
                decompiler_cache={}, comparator_cache={}, joint_cache={},
            )
        audit.assert_not_called()
        self.assertEqual(check_blueprint.call_count, 1)
        self.assertEqual(check_blueprint.call_args.args[0], _blueprint.lean_file)
        self.assertIn("sorry_using [model]", _blueprint.lean_file)
        self.assertEqual(validation.mechanical_failure_stage, "canonical_lean")
        self.assertTrue(deterministic)
        self.assertFalse(semantic)

    def test_eligible_round_compiles_whole_formal_blueprint_once(self) -> None:
        check_blueprint = Mock(return_value=CompilerResult(True))
        compiler = SimpleNamespace(check_blueprint=check_blueprint)
        standalone = Phase2StandaloneReport((), 2, 0, 1.0)
        with (
            patch(
                "blueprint_generation.phase2_standalone_contract_report",
                return_value=standalone,
            ),
            patch(
                "blueprint_generation._with_semantic_audit",
                side_effect=lambda validation, *_args, **_kwargs: validation,
            ),
        ):
            blueprint, validation, deterministic, semantic, _warnings = _validate_round(
                self.MINIMAL, target_name="root", compiler=compiler,
                informal_statement="problem", informal_proof="cot", claimed_answer="1",
                standalone_concurrency=1, client=object(), model="model",
                decompiler_max_tokens=16, comparator_max_tokens=16,
                semantic_format_attempts=2, semantic_audit_enable_thinking=True,
                semantic_audit_temperature=0.0, semantic_audit_top_p=0.95,
                semantic_audit_top_k=20, semantic_audit_min_p=0.0,
                semantic_audit_presence_penalty=0.0,
                semantic_audit_repetition_penalty=1.0,
                semantic_audit_mode="direct", node_naming="semantic",
                joint_semantic_audit_max_tokens=32768,
                tokenizer_path="unused", model_max_context=40960,
                context_safety_margin=512, tracer=None,
                thm_name="id", round_index=1, standalone_cache={},
                decompiler_cache={}, comparator_cache={}, joint_cache={},
            )
        self.assertEqual(check_blueprint.call_count, 1)
        self.assertEqual(check_blueprint.call_args.args[0], blueprint.lean_file)
        self.assertIs(validation.lean_result, validation.canonical_lean_result)
        self.assertFalse(deterministic)
        self.assertFalse(semantic)

    def test_eligible_round_request_counts_differ_by_audit_mode(self) -> None:
        blueprint = _parse_blueprint(self.MINIMAL, "root")
        base = BlueprintValidation(
            lean_result=CompilerResult(True),
            canonical_lean_result=CompilerResult(True),
            semantic_issues=[], structural_errors=[],
            standalone_report=Phase2StandaloneReport((), 2, 0, 1.0),
            mechanical_stage_reached="static_shadow",
        )
        decompiler = FormalDecompilerResult(
            (), "{}", "", "stop", "d", 1, 1, 2,
            ({"attempt": 1},),
        )
        comparator = WholeCotComparatorResult(
            {
                "combined_formal_translation": "x", "missing_clauses": [],
                "weakened_clauses": [], "unbound_objects": [],
                "wrong_relations": [], "added_clauses": [],
            },
            {
                "translation": "x", "target_object_preserved": True,
                "answer_grounded": True, "reasons": [],
            },
            (), (), True, "{}", "", "stop", "c", 1, 1, 2,
            ({"attempt": 1},),
        )
        common = dict(
            client=object(), model="model", informal_statement="problem",
            informal_proof="cot", claimed_answer="1",
            formal_decompiler_max_tokens=16384,
            strict_comparator_max_tokens=16384, format_max_attempts=2,
            semantic_audit_enable_thinking=True,
            semantic_audit_temperature=0.6, semantic_audit_top_p=0.95,
            semantic_audit_top_k=20, semantic_audit_min_p=0.0,
            semantic_audit_presence_penalty=0.0,
            semantic_audit_repetition_penalty=1.0,
            joint_semantic_audit_max_tokens=32768,
            tokenizer_path="unused", model_max_context=40960,
            context_safety_margin=512, tracer=None, thm_name="id",
            round_index=1,
        )
        with (
            patch("blueprint_generation.run_formal_decompiler", return_value=decompiler) as run_d,
            patch("blueprint_generation.run_whole_cot_comparator", return_value=comparator) as run_c,
        ):
            separate = _with_semantic_audit(
                base, blueprint, semantic_audit_mode="separate",
                decompiler_cache={}, comparator_cache={}, joint_cache={}, **common,
            )
        self.assertEqual(separate.semantic_request_count, 2)
        run_d.assert_called_once()
        run_c.assert_called_once()

        with (
            patch(
                "blueprint_generation.run_direct_whole_cot_comparator",
                return_value=comparator,
            ) as run_direct,
        ):
            direct = _with_semantic_audit(
                base, blueprint, semantic_audit_mode="direct",
                decompiler_cache={}, comparator_cache={}, joint_cache={}, **common,
            )
        self.assertEqual(direct.semantic_request_count, 1)
        self.assertIsNone(direct.formal_decompiler_result)
        run_direct.assert_called_once()

        joint_value = JointWholeCotAuditResult(
            decompiler, comparator, "{}", "", "stop", "j", 1, 1, 2,
            ({"attempt": 1},),
        )
        joint_cache = {}
        with (
            patch("blueprint_generation.generation_request_budget", return_value=(100, 40000)),
            patch("blueprint_generation.run_joint_whole_cot_audit", return_value=joint_value) as run_j,
        ):
            joint = _with_semantic_audit(
                base, blueprint, semantic_audit_mode="joint",
                decompiler_cache={}, comparator_cache={}, joint_cache=joint_cache,
                **common,
            )
            cached = _with_semantic_audit(
                base, blueprint, semantic_audit_mode="joint",
                decompiler_cache={}, comparator_cache={}, joint_cache=joint_cache,
                **common,
            )
        self.assertEqual(joint.semantic_request_count, 1)
        self.assertEqual(joint.semantic_output_budget, 32768)
        self.assertEqual(cached.semantic_request_count, 0)
        self.assertTrue(cached.semantic_cache_hits["joint"])
        run_j.assert_called_once()

    def test_joint_schema_exhaustion_preserves_terminal_request_count(self) -> None:
        blueprint = _parse_blueprint(self.MINIMAL, "root")
        validation = BlueprintValidation(
            lean_result=CompilerResult(True),
            canonical_lean_result=CompilerResult(True),
            semantic_issues=[], structural_errors=[],
            standalone_report=Phase2StandaloneReport((), 2, 0, 1.0),
            mechanical_stage_reached="static_shadow",
        )
        error = SemanticAuditFormatError(
            "bad joint schema", attempts=({"attempt": 1}, {"attempt": 2}),
        )
        with (
            patch("blueprint_generation.generation_request_budget", return_value=(100, 40000)),
            patch("blueprint_generation.run_joint_whole_cot_audit", side_effect=error),
            self.assertRaises(SemanticAuditFormatError),
        ):
            _with_semantic_audit(
                validation, blueprint, client=object(), model="model",
                informal_statement="problem", informal_proof="cot",
                claimed_answer="1", formal_decompiler_max_tokens=16384,
                strict_comparator_max_tokens=16384, format_max_attempts=2,
                semantic_audit_enable_thinking=True,
                semantic_audit_temperature=0.6, semantic_audit_top_p=0.95,
                semantic_audit_top_k=20, semantic_audit_min_p=0.0,
                semantic_audit_presence_penalty=0.0,
                semantic_audit_repetition_penalty=1.0,
                semantic_audit_mode="joint",
                joint_semantic_audit_max_tokens=32768,
                tokenizer_path="unused", model_max_context=40960,
                context_safety_margin=512, decompiler_cache={},
                comparator_cache={}, joint_cache={}, tracer=None,
                thm_name="id", round_index=1,
            )
        self.assertTrue(validation.semantic_audit_invoked)
        self.assertEqual(validation.semantic_audit_mode, "joint")
        self.assertEqual(validation.semantic_request_count, 2)
        self.assertEqual(validation.semantic_output_budget, 32768)

    def test_semantic_audit_failure_is_terminal_without_regeneration(self) -> None:
        validation = BlueprintValidation(
            lean_result=CompilerResult(True),
            canonical_lean_result=CompilerResult(True),
            semantic_issues=[], structural_errors=[],
            standalone_report=Phase2StandaloneReport((), 2, 0, 1.0),
            mechanical_stage_reached="joint_semantic_audit",
            semantic_audit_invoked=True, semantic_audit_mode="joint",
            semantic_request_count=2, semantic_output_budget=32768,
        )
        from blueprint_generation import SemanticAuditExecutionError
        formal_blueprint = _parse_blueprint(self.MINIMAL, "root")
        with (
            patch("blueprint_generation.make_client", return_value=object()),
            patch("blueprint_generation.generation_request_budget", return_value=(100, 1000)),
            patch("blueprint_generation.chat_completion_with_retry", return_value=object()) as generate,
            patch("blueprint_generation._emit_usage"),
            patch("blueprint_generation._emit_llm_response"),
            patch("blueprint_generation._submitted_code", return_value=(self.MINIMAL, ())),
            patch(
                "blueprint_generation._validate_round",
                side_effect=SemanticAuditExecutionError(
                    "schema failed", validation, formal_blueprint,
                ),
            ),
            self.assertRaises(BlueprintGenerationError) as caught,
        ):
            generate_blueprint(
                informal_statement="problem", informal_proof="cot",
                claimed_answer="1", target_name="root", model="model",
                compiler=object(), tracer=None, thm_name="id", max_turns=8,
                tokenizer_path="unused", model_max_context=40960,
                context_safety_margin=512, enable_thinking=True,
                temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                presence_penalty=0.0, repetition_penalty=1.0,
                standalone_concurrency=1, decompiler_max_tokens=16384,
                comparator_max_tokens=16384, semantic_format_attempts=2,
                semantic_audit_enable_thinking=True,
                semantic_audit_temperature=0.6, semantic_audit_top_p=0.95,
                semantic_audit_top_k=20, semantic_audit_min_p=0.0,
                semantic_audit_presence_penalty=0.0,
                semantic_audit_repetition_penalty=1.0,
                semantic_audit_mode="joint",
                joint_semantic_audit_max_tokens=32768,
            )
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(
            caught.exception.validation_details["classification"],
            "semanticAuditError",
        )
        self.assertEqual(
            caught.exception.validation_details["semanticActualRequestCount"], 2,
        )
        self.assertEqual(caught.exception.last_candidate, formal_blueprint.lean_file)
        self.assertEqual(
            caught.exception.candidate_history[-1], formal_blueprint.lean_file,
        )

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

    def test_anonymous_generation_prompt_and_contract(self) -> None:
        messages = _messages(
            target_name="n_final", informal_statement="problem",
            informal_proof="cot", claimed_answer="6",
            previous_blueprint="", previous_feedback="",
            prompt_profile="whole_cot_minimal", node_naming="anonymous",
        )
        rendered = "\n".join(item["content"] for item in messages)
        self.assertIn("`d1`, `d2`", rendered)
        self.assertIn("`n_final`", rendered)
        valid = _parse_blueprint('''import Mathlib
import Architect
@[blueprint] def d1 : Nat := 6
@[blueprint] lemma n1 : d1 = 6 := by sorry_using [d1]
@[blueprint] theorem n_final : d1 = 6 := by sorry_using [d1, n1]
''', "n_final")
        self.assertEqual(
            _contract_errors(valid, "n_final", node_naming="anonymous"), [],
        )
        invalid = _parse_blueprint(self.MINIMAL, "root")
        codes = {
            item["code"] for item in _contract_errors(
                invalid, "root", node_naming="anonymous",
            )
        }
        self.assertIn("anonymousDefinitionNames", codes)
        self.assertIn("anonymousRootName", codes)


if __name__ == "__main__":
    unittest.main()
