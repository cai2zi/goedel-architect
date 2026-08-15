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
    _eligible_mathlib_symbols,
    _feedback,
    _run_phase1_mathlib_search,
    _semantic_feedback_state,
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

    def _generation_kwargs(self, *, max_semantic_audits: int = 8) -> dict:
        return {
            "informal_statement": "problem", "informal_proof": "cot",
            "claimed_answer": "1", "target_name": "root", "model": "model",
            "compiler": object(), "tracer": None, "thm_name": "id",
            "max_semantic_audits": max_semantic_audits,
            "tokenizer_path": "unused", "model_max_context": 40960,
            "context_safety_margin": 512, "enable_thinking": True,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0,
            "standalone_concurrency": 1, "decompiler_max_tokens": 16384,
            "comparator_max_tokens": 16384, "semantic_format_attempts": 2,
            "semantic_audit_enable_thinking": True,
            "semantic_audit_temperature": 0.0, "semantic_audit_top_p": 0.95,
            "semantic_audit_top_k": 20, "semantic_audit_min_p": 0.0,
            "semantic_audit_presence_penalty": 0.0,
            "semantic_audit_repetition_penalty": 1.0,
            "semantic_audit_mode": "direct",
            "semantic_comparator_protocol": "canonical_v2",
            "joint_semantic_audit_max_tokens": 32768,
        }

    @staticmethod
    def _generation_response(content: str = "") -> SimpleNamespace:
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="tool_calls",
            message=SimpleNamespace(content=content, tool_calls=[]),
        )])

    @staticmethod
    def _mock_validation(*, semantic_invoked: bool, structural: bool = False) -> BlueprintValidation:
        return BlueprintValidation(
            lean_result=CompilerResult(not structural),
            canonical_lean_result=CompilerResult(not structural),
            semantic_issues=[], structural_errors=[],
            standalone_report=Phase2StandaloneReport((), 2, 0, 1.0),
            mechanical_stage_reached="canonical_lean" if structural else "static_shadow",
            mechanical_failure_stage="canonical_lean" if structural else None,
            semantic_audit_invoked=semantic_invoked,
            semantic_audit_mode="direct",
        )

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

    def test_generation_prompt_contains_mechanical_and_coordinate_contract(self) -> None:
        messages = _messages(
            target_name="root", informal_statement="problem", informal_proof="cot",
            claimed_answer="1", previous_blueprint="", previous_feedback="",
        )
        system = messages[0]["content"]
        self.assertIn("∑ x ∈ s, f x", system)
        self.assertIn("Never use\n`∑ x in s", system)
        self.assertIn("structure", system)
        self.assertIn("tuple or type alias plus accessor", system)
        self.assertIn("noncomputable def", system)
        self.assertIn("max/sup metric, not the Euclidean metric", system)
        self.assertIn("squared Euclidean distance", system)

    def test_answer_preassigned_guidance_is_dynamic_and_synthetic(self) -> None:
        common = dict(
            target_name="root", informal_statement="problem", informal_proof="cot",
            claimed_answer="1", previous_blueprint=self.MINIMAL,
            previous_feedback="semantic defect",
        )
        plain = "\n".join(item["content"] for item in _messages(**common))
        coverage = "\n".join(item["content"] for item in _messages(
            **common, active_repair_codes=("targetCoverageIncomplete",),
        ))
        repaired = "\n".join(item["content"] for item in _messages(
            **common, active_repair_codes=("answerPreassigned",),
        ))
        self.assertNotIn("demo_relations", plain)
        self.assertNotIn("demo_relations", coverage)
        self.assertIn("demo_relations", repaired)
        self.assertIn("theorem root", repaired)
        for forbidden in ("Wrong76", "prealgebra/874", "26", "115"):
            self.assertNotIn(forbidden, repaired)

    def test_semantic_feedback_retains_across_mechanical_failures_and_replaces(self) -> None:
        first = ({"code": "answerPreassigned", "message": "bind unknown"},)
        errors, source, codes, retained = _semantic_feedback_state(
            (), None, semantic_audit_invoked=True, current_errors=first,
            current_round=1,
        )
        self.assertEqual(errors, first)
        self.assertEqual(source, 1)
        self.assertEqual(codes, ("answerPreassigned",))
        self.assertFalse(retained)
        for round_index in (2, 3):
            errors, source, codes, retained = _semantic_feedback_state(
                errors, source, semantic_audit_invoked=False, current_errors=(),
                current_round=round_index,
            )
            self.assertEqual(errors, first)
            self.assertEqual(source, 1)
            self.assertEqual(codes, ("answerPreassigned",))
            self.assertTrue(retained)
        replacement = ({"code": "targetCoverageIncomplete", "message": "all roots"},)
        errors, source, codes, retained = _semantic_feedback_state(
            errors, source, semantic_audit_invoked=True,
            current_errors=replacement, current_round=4,
        )
        self.assertEqual(errors, replacement)
        self.assertEqual(source, 4)
        self.assertEqual(codes, ("targetCoverageIncomplete",))
        self.assertFalse(retained)

    def test_new_audit_without_answer_preassigned_stops_guidance(self) -> None:
        errors, source, codes, _ = _semantic_feedback_state(
            ({"code": "answerPreassigned"},), 1,
            semantic_audit_invoked=True, current_errors=(), current_round=2,
        )
        self.assertEqual(errors, ())
        self.assertEqual(source, 2)
        self.assertEqual(codes, ())

    def test_narrow_contract_feedback_has_only_validated_guidance(self) -> None:
        blueprint = _parse_blueprint(self.MINIMAL, "root")
        root_line = blueprint.nodes_by_name()["root"].lean_start_line
        explicit_noncomputable = json.dumps({
            "severity": "error",
            "pos": {"line": root_line + 1, "column": 1},
            "data": "failed to compile definition, consider marking it as 'noncomputable'",
        })
        deterministic = [
            {"stage": "canonical_lean", "code": "canonicalLean", "nodeName": "",
             "message": "Safeguard rejected: forbidden construct `structure` is not allowed."},
            {"stage": "parse_basic", "code": "unannotatedLocalDeclaration", "nodeName": "",
             "message": "unannotatedLocalDeclaration: x"},
            {"stage": "canonical_lean", "code": "canonicalLean", "nodeName": "",
             "message": explicit_noncomputable},
            {"stage": "canonical_lean", "code": "canonicalLean", "nodeName": "",
             "message": '{"data":"type mismatch","pos":{"line":1,"column":1}}'},
        ]
        feedback = _feedback(deterministic, (), (), blueprint=blueprint)
        context = feedback.split("VALIDATED_CONTRACT_CONTEXT", 1)[1]
        self.assertIn("tuple or type alias", context)
        self.assertIn("Canonicalization keeps only imports", context)
        self.assertIn("node `root`", context)
        self.assertNotIn("type mismatch", context)

    def test_malformed_or_unmapped_noncomputable_diagnostic_adds_no_context(self) -> None:
        blueprint = _parse_blueprint(self.MINIMAL, "root")
        for message in (
            "consider marking it as 'noncomputable'",
            json.dumps({"data": "consider marking it as 'noncomputable'",
                        "pos": {"line": 999, "column": 1}}),
        ):
            with self.subTest(message=message):
                feedback = _feedback([{
                    "stage": "canonical_lean", "code": "canonicalLean",
                    "nodeName": "", "message": message,
                }], (), (), blueprint=blueprint)
                self.assertNotIn("VALIDATED_CONTRACT_CONTEXT", feedback)

    def test_mathlib_retrieval_is_narrow_bounded_cached_and_non_mutating(self) -> None:
        class FakeRetrieval:
            def __init__(self):
                self.calls = []

            def search(self, query, k):
                self.calls.append((query, k))
                return [
                    {"name": f"{query}.candidate{i}", "type": f"Type{i}"}
                    for i in range(5)
                ]

        errors = [
            {"stage": "canonical_lean", "code": "canonicalLean",
             "message": '{"data":"Unknown constant `Complex.abs`"}'},
            {"stage": "canonical_lean", "code": "canonicalLean",
             "message": '{"data":"Invalid field `List.get!`"}'},
            {"stage": "canonical_lean", "code": "canonicalLean",
             "message": '{"data":"Unknown identifier `third.symbol`"}'},
            {"stage": "phase2_contract", "code": "unknownDependencies",
             "message": "Unknown identifier `must.not.search`"},
            {"stage": "canonical_lean", "code": "canonicalLean",
             "message": '{"data":"type mismatch"}'},
        ]
        retrieval = FakeRetrieval()
        cache = {}
        first = _run_phase1_mathlib_search(
            errors, retrieval=retrieval, cache=cache, max_queries=2, k=3,
        )
        second = _run_phase1_mathlib_search(
            errors, retrieval=retrieval, cache=cache, max_queries=2, k=3,
        )
        self.assertEqual(
            _eligible_mathlib_symbols(errors),
            ("Complex.abs", "List.get!", "third.symbol"),
        )
        self.assertEqual(retrieval.calls, [("Complex.abs", 3), ("List.get!", 3)])
        self.assertTrue(all(len(item["results"]) == 3 for item in first))
        self.assertTrue(all(item["cacheHit"] for item in second))
        feedback = _feedback((), (), (), mathlib_search_context=first)
        self.assertIn("candidates, not a repair conclusion", feedback)
        self.assertIn("never add a Mathlib name to `sorry_using`", feedback)

    def test_mathlib_retrieval_failure_degrades_to_empty_context(self) -> None:
        retrieval = SimpleNamespace(search=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TimeoutError("offline")
        ))
        reports = _run_phase1_mathlib_search([{
            "stage": "canonical_lean", "code": "canonicalLean",
            "message": '{"data":"Unknown identifier `missingApi`"}',
        }], retrieval=retrieval, cache={}, max_queries=2, k=3)
        self.assertEqual(reports[0]["results"], [])
        self.assertIn("TimeoutError", reports[0]["error"])

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
                semantic_comparator_protocol="legacy_v1",
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
                semantic_comparator_protocol="legacy_v1",
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
            (), (), (), True, "{}", "", "stop", "c", 1, 1, 2,
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
                semantic_comparator_protocol="legacy_v1",
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
                semantic_comparator_protocol="legacy_v1",
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
                semantic_comparator_protocol="legacy_v1",
                decompiler_cache={}, comparator_cache={}, joint_cache=joint_cache,
                **common,
            )
            cached = _with_semantic_audit(
                base, blueprint, semantic_audit_mode="joint",
                semantic_comparator_protocol="legacy_v1",
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
                semantic_comparator_protocol="legacy_v1",
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
            patch(
                "blueprint_generation.chat_completion_with_retry",
                return_value=self._generation_response(),
            ) as generate,
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
                compiler=object(), tracer=None, thm_name="id", max_semantic_audits=8,
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

    def test_structural_retries_preserve_semantic_anchor_and_do_not_consume_budget(self) -> None:
        codes = [
            self.MINIMAL.replace("model : Nat := 1", f"model : Nat := {value}")
            for value in (1, 2, 3, 4)
        ]
        blueprints = [_parse_blueprint(code, "root") for code in codes]
        semantic_error = ({
            "stage": "whole_cot_comparator", "code": "derivationShortcut",
            "nodeName": "root", "message": "derive the unknown",
        },)
        deterministic_2 = ({
            "stage": "canonical_lean", "code": "canonicalLean",
            "nodeName": "", "message": "unknown declaration two",
        },)
        deterministic_3 = ({
            "stage": "phase2_standalone", "code": "phase2StandaloneFailed",
            "nodeName": "root", "message": "unknown declaration three",
        },)
        validations = [
            (blueprints[0], self._mock_validation(semantic_invoked=True), (), semantic_error, ()),
            (blueprints[1], self._mock_validation(semantic_invoked=False, structural=True), deterministic_2, (), ()),
            (blueprints[2], self._mock_validation(semantic_invoked=False, structural=True), deterministic_3, (), ()),
            (blueprints[3], self._mock_validation(semantic_invoked=True), (), (), ()),
        ]
        with (
            patch("blueprint_generation.make_client", return_value=object()),
            patch("blueprint_generation.generation_request_budget", return_value=(100, 1000)),
            patch(
                "blueprint_generation.chat_completion_with_retry",
                side_effect=[self._generation_response() for _ in codes],
            ) as generate,
            patch("blueprint_generation._emit_usage"),
            patch("blueprint_generation._emit_llm_response"),
            patch("blueprint_generation._submitted_code", side_effect=[(code, ()) for code in codes]),
            patch("blueprint_generation._validate_round", side_effect=validations),
        ):
            accepted = generate_blueprint(**self._generation_kwargs(max_semantic_audits=2))

        self.assertEqual(generate.call_count, 4)
        history = accepted.generation_history
        self.assertEqual([item["semanticStage"] for item in history], [1, 2, 2, 2])
        self.assertEqual(
            [item["semanticAuditOrdinal"] for item in history], [1, None, None, 2],
        )
        self.assertEqual(
            [item["semanticAnchorRound"] for item in history], [None, 1, 1, 1],
        )
        self.assertEqual(
            [item["structuralInputRound"] for item in history], [None, None, 2, 3],
        )
        self.assertEqual(
            [item["attemptRole"] for item in history],
            ["semanticAudited", "structuralRetry", "structuralRetry", "semanticAudited"],
        )
        self.assertEqual(accepted.generation_validation["semanticAuditCount"], 2)
        self.assertIn("generation_round_2_submitted", accepted.candidate_labels)
        self.assertIn("generation_round_4_canonical", accepted.candidate_labels)

    def test_eighth_semantic_reject_terminates_with_audited_anchor(self) -> None:
        blueprint = _parse_blueprint(self.MINIMAL, "root")
        semantic_error = ({
            "stage": "whole_cot_comparator", "code": "targetMismatch",
            "nodeName": "root", "message": "wrong target",
        },)
        with (
            patch("blueprint_generation.make_client", return_value=object()),
            patch("blueprint_generation.generation_request_budget", return_value=(100, 1000)),
            patch(
                "blueprint_generation.chat_completion_with_retry",
                side_effect=[self._generation_response() for _ in range(8)],
            ),
            patch("blueprint_generation._emit_usage"),
            patch("blueprint_generation._emit_llm_response"),
            patch(
                "blueprint_generation._submitted_code",
                side_effect=[(self.MINIMAL, ()) for _ in range(8)],
            ),
            patch(
                "blueprint_generation._validate_round",
                side_effect=[(
                    blueprint, self._mock_validation(semantic_invoked=True),
                    (), semantic_error, (),
                ) for _ in range(8)],
            ),
            self.assertRaises(BlueprintGenerationError) as caught,
        ):
            generate_blueprint(**self._generation_kwargs(max_semantic_audits=8))
        error = caught.exception
        self.assertEqual(error.validation_details["classification"], "semanticRejected")
        self.assertEqual(error.validation_details["semanticAuditCount"], 8)
        self.assertEqual(error.last_candidate, blueprint.lean_file)
        self.assertEqual(error.generation_history[-1]["semanticAuditOrdinal"], 8)
        self.assertFalse(error.validation_details["finalDeterministicErrors"])

    def test_context_fallback_omits_only_latest_structural_body(self) -> None:
        anchor_code = self.MINIMAL
        structural_code = self.MINIMAL.replace("model : Nat := 1", "model : Nat := 2")
        accepted_code = self.MINIMAL.replace("model : Nat := 1", "model : Nat := 3")
        anchor = _parse_blueprint(anchor_code, "root")
        structural = _parse_blueprint(structural_code, "root")
        accepted = _parse_blueprint(accepted_code, "root")
        semantic_error = ({
            "stage": "whole_cot_comparator", "code": "answerShortcut",
            "nodeName": "root", "message": "retain this semantic issue",
        },)
        structural_error = ({
            "stage": "canonical_lean", "code": "canonicalLean",
            "nodeName": "", "message": "repair this structure",
        },)

        def budget(messages, **_kwargs):
            rendered = "\n".join(message["content"] for message in messages)
            if "Latest structurally rejected Blueprint" in rendered:
                return 50000, -100
            return 100, 1000

        validations = [
            (anchor, self._mock_validation(semantic_invoked=True), (), semantic_error, ()),
            (structural, self._mock_validation(semantic_invoked=False, structural=True), structural_error, (), ()),
            (accepted, self._mock_validation(semantic_invoked=True), (), (), ()),
        ]
        with (
            patch("blueprint_generation.make_client", return_value=object()),
            patch("blueprint_generation.generation_request_budget", side_effect=budget),
            patch(
                "blueprint_generation.chat_completion_with_retry",
                side_effect=[self._generation_response() for _ in range(3)],
            ) as generate,
            patch("blueprint_generation._emit_usage"),
            patch("blueprint_generation._emit_llm_response"),
            patch(
                "blueprint_generation._submitted_code",
                side_effect=[(anchor_code, ()), (structural_code, ()), (accepted_code, ())],
            ),
            patch("blueprint_generation._validate_round", side_effect=validations),
        ):
            result = generate_blueprint(**self._generation_kwargs(max_semantic_audits=2))
        third = result.generation_history[2]
        self.assertTrue(third["contextFallbackApplied"])
        rendered = "\n".join(
            item["content"] for item in generate.call_args_list[2].kwargs["messages"]
        )
        self.assertIn("Last semantically audited Blueprint", rendered)
        self.assertIn("retain this semantic issue", rendered)
        self.assertIn("repair this structure", rendered)
        self.assertNotIn("Latest structurally rejected Blueprint\n```lean", rendered)
        self.assertIn("was omitted because", rendered)

    def test_invalid_tool_submission_is_an_unbudgeted_structural_retry(self) -> None:
        anchor = _parse_blueprint(self.MINIMAL, "root")
        accepted_code = self.MINIMAL.replace("model : Nat := 1", "model : Nat := 4")
        accepted = _parse_blueprint(accepted_code, "root")
        semantic_error = ({
            "stage": "whole_cot_comparator", "code": "semanticMismatch",
            "nodeName": "root", "message": "retain semantic anchor",
        },)
        submission_error = ({
            "stage": "parse_basic", "code": "phase1ToolCallCount",
            "nodeName": "", "message": "expected one tool call",
        },)
        with (
            patch("blueprint_generation.make_client", return_value=object()),
            patch("blueprint_generation.generation_request_budget", return_value=(100, 1000)),
            patch(
                "blueprint_generation.chat_completion_with_retry",
                side_effect=[self._generation_response() for _ in range(3)],
            ) as generate,
            patch("blueprint_generation._emit_usage"),
            patch("blueprint_generation._emit_llm_response"),
            patch(
                "blueprint_generation._submitted_code",
                side_effect=[(self.MINIMAL, ()), ("", submission_error), (accepted_code, ())],
            ),
            patch("blueprint_generation._validate_round", side_effect=[
                (anchor, self._mock_validation(semantic_invoked=True), (), semantic_error, ()),
                (accepted, self._mock_validation(semantic_invoked=True), (), (), ()),
            ]),
        ):
            result = generate_blueprint(**self._generation_kwargs(max_semantic_audits=2))
        history = result.generation_history
        self.assertEqual([item["semanticAuditOrdinal"] for item in history], [1, None, 2])
        self.assertEqual(history[1]["attemptRole"], "structuralRetry")
        self.assertEqual(history[2]["semanticAnchorRound"], 1)
        self.assertEqual(history[2]["structuralInputRound"], 2)
        rendered = "\n".join(
            item["content"] for item in generate.call_args_list[2].kwargs["messages"]
        )
        self.assertIn("retain semantic anchor", rendered)
        self.assertIn("expected one tool call", rendered)
        self.assertNotIn("Latest structurally rejected Blueprint\n```lean", rendered)

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
        self.assertIsNone(generation_round_classification(
            round_index=8, max_turns=8, deterministic_error_count=1,
            semantic_error_count=2, warning_count=0,
        ))
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
