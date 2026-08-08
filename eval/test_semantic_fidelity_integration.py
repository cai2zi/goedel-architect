from __future__ import annotations

import hashlib
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blueprint import BlueprintGenerationError, _parse_blueprint, generate_blueprint_from_informal  # noqa: E402
from kimina_lean_compiler import CompilerResult  # noqa: E402
from orchestrator import OrchestratorResult  # noqa: E402
from refinement import SemanticRefinementError, refine_blueprint  # noqa: E402
from semantic_audit import SemanticAuditResult  # noqa: E402
from semantic_fidelity import SemanticIssue  # noqa: E402


def manifest() -> str:
    rows = []
    for index, (text, role) in enumerate((
        ("Set up x.", "setup"),
        ("Claim x is 45.", "derived_claim"),
        ("Conclude 45.", "conclusion"),
    ), 1):
        rows.append({
            "step_id": f"S{index:03d}",
            "source_text": text,
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "role": role,
            "depends_on": [f"S{index - 1:03d}"] if index > 1 else [],
            "numbers": ["45"] if index > 1 else [],
            "relations": ["eq"] if index > 1 else [],
        })
    return json.dumps(rows)


def claim_manifest() -> str:
    rows = json.loads(manifest())
    for row in rows:
        claim_id = f"{row['step_id']}.C001"
        row["claims"] = [{
            "claim_id": claim_id,
            "source_text": row["source_text"],
            "source_sha256": hashlib.sha256(row["source_text"].encode()).hexdigest(),
        }]
    return json.dumps(rows)


def response(lean: str):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=f"```lean\n{lean}\n```", tool_calls=[]),
        finish_reason="stop",
    )], usage=None)


def baseline_code() -> str:
    return """import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- setup -/)]
def quantity : ℕ := 40
@[blueprint (title := "COT_STEP:S002") (statement := /-- claim -/) (proof := /-- source -/)]
lemma calculated : quantity + 5 = 45 := by sorry_using [quantity]
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : quantity + 5 = 45 := by sorry_using [calculated]
"""


class NeverCompiler:
    def __init__(self) -> None:
        self.calls = 0

    def check_blueprint(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("semantic rejection must happen before Lean")


class PassingCompiler:
    def check_blueprint(self, *_args, **_kwargs):
        return CompilerResult(True)

    def check_many(self, requests, **_kwargs):
        return [CompilerResult(True) for _request in requests]


class CapturingCompiler(PassingCompiler):
    def __init__(self) -> None:
        self.blueprints: list[str] = []

    def check_blueprint(self, lean_code, *_args, **_kwargs):
        self.blueprints.append(lean_code)
        return CompilerResult(True)


def passing_audit() -> SemanticAuditResult:
    return SemanticAuditResult(
        passed=True,
        flag="PASS",
        diagnostics="",
        raw_content="[[SEMANTIC_AUDIT=PASS]]",
        reasoning_content="",
        model="audit-model",
        mode="full",
        finish_reason="stop",
        truncated=False,
        request_id="audit-request",
        prompt_tokens=100,
        completion_tokens=5,
        total_tokens=105,
    )


def failing_audit() -> SemanticAuditResult:
    return SemanticAuditResult(
        passed=False,
        flag="FAIL",
        diagnostics="S002: The formal proposition drops the source equality.",
        raw_content=(
            "[[SEMANTIC_AUDIT=FAIL]]\n"
            "S002: The formal proposition drops the source equality."
        ),
        reasoning_content="",
        model="audit-model",
        mode="full",
        finish_reason="stop",
        truncated=False,
        request_id="audit-request",
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
    )


class SemanticFidelityIntegrationTest(unittest.TestCase):
    def test_phase1_infers_formal_reachability_without_model_retry(self) -> None:
        disconnected = """import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001.C001") (statement := /-- setup -/)]
def quantity : ℕ := 40
@[blueprint (title := "COT_STEP:S002.C001") (statement := /-- claim -/) (proof := /-- source -/)]
lemma calculated : quantity + 5 = 45 := by sorry_using []
@[blueprint (title := "COT_STEP:S003.C001") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : quantity + 5 = 45 := by sorry_using [calculated]
"""
        compiler = CapturingCompiler()
        with patch("blueprint.make_client", return_value=object()), patch(
            "blueprint._call_blueprint_model", return_value=response(disconnected),
        ) as call_model:
            generated = generate_blueprint_from_informal(
                "Find the original quantity plus five.",
                "The COT concludes 45.",
                "target",
                model="Qwen3.5-test",
                compiler=compiler,
                cot_manifest_json=claim_manifest(),
                claimed_answer="45",
                semantic_fidelity_enabled=True,
                semantic_require_step_ids=True,
                semantic_static_gate=True,
                semantic_audit_mode="none",
                semantic_max_repair_attempts=0,
                max_retries=1,
            )

        self.assertEqual(call_model.call_count, 1)
        self.assertEqual(len(compiler.blueprints), 1)
        self.assertIn(
            "theorem target : quantity + 5 = 45 := by sorry_using [calculated]",
            compiler.blueprints[0],
        )
        self.assertEqual(len(generated.semantic_gate_results), 1)
        self.assertEqual(generated.semantic_gate_results[0]["stage"], "phase1_local_gate")

    def test_e1_reachability_repair_ignores_disabled_static_issues(self) -> None:
        disconnected_with_hidden_static_issue = """import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S001") (statement := /-- setup -/)]
def vacuous_setup : Prop := True
@[blueprint (title := "COT_STEP:S002") (statement := /-- claim -/) (proof := /-- source -/)]
lemma calculated : 40 + 5 = 45 := by sorry_using []
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : 40 + 5 = 45 := by sorry_using [calculated]
"""
        compiler = CapturingCompiler()
        with patch("blueprint.make_client", return_value=object()), patch(
            "blueprint._call_blueprint_model",
            return_value=response(disconnected_with_hidden_static_issue),
        ) as call_model:
            generated = generate_blueprint_from_informal(
                "Find the original quantity plus five.",
                "The COT concludes 45.",
                "target",
                model="Qwen3.5-test",
                compiler=compiler,
                cot_manifest_json=manifest(),
                claimed_answer="45",
                semantic_fidelity_enabled=True,
                semantic_require_step_ids=True,
                semantic_static_gate=False,
                semantic_audit_mode="none",
                semantic_max_repair_attempts=0,
                max_retries=1,
            )

        self.assertEqual(call_model.call_count, 1)
        self.assertEqual(len(compiler.blueprints), 1)
        self.assertIn(
            "theorem target : 40 + 5 = 45 := by sorry_using "
            "[calculated, vacuous_setup]",
            compiler.blueprints[0],
        )
        graph_events = [
            event for event in generated.semantic_gate_results
            if event["stage"] == "phase1_graph_repair"
        ]
        self.assertEqual(len(graph_events), 1)
        self.assertEqual(graph_events[0]["added_dependencies"], ["vacuous_setup"])

    def test_phase1_local_and_audit_repairs_have_independent_budgets(self) -> None:
        local_issue = SemanticIssue(
            code="VACUOUS_TRUE_STEP",
            message="synthetic local mismatch",
            node_name="calculated",
            step_id="S002",
        )
        with patch("blueprint.make_client", return_value=object()), patch(
            "blueprint._call_blueprint_model", return_value=response(baseline_code()),
        ) as call_model, patch(
            "blueprint.validate_blueprint_fidelity",
            side_effect=[[local_issue], [], []],
        ), patch(
            "blueprint.run_semantic_audit",
            side_effect=[failing_audit(), passing_audit()],
        ) as audit:
            generated = generate_blueprint_from_informal(
                "Find the original quantity plus five.",
                "The COT concludes 45.",
                "target",
                model="Qwen3.5-test",
                compiler=PassingCompiler(),
                cot_manifest_json=manifest(),
                claimed_answer="45",
                semantic_fidelity_enabled=True,
                semantic_require_step_ids=True,
                semantic_static_gate=True,
                semantic_audit_mode="full",
                semantic_max_repair_attempts=1,
                max_retries=3,
            )

        self.assertEqual(generated.target_theorem, "target")
        self.assertEqual(call_model.call_count, 3)
        self.assertEqual(audit.call_count, 2)
        first_kwargs = call_model.call_args_list[0].kwargs["reasoning_kwargs"]
        repair_kwargs = call_model.call_args_list[1].kwargs["reasoning_kwargs"]
        self.assertNotIn("temperature", first_kwargs)
        self.assertEqual(repair_kwargs["temperature"], 0)
        self.assertEqual(
            repair_kwargs["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_phase1_audit_receives_original_problem_and_claimed_answer(self) -> None:
        audit_calls = []

        def audit(*args, **kwargs):
            audit_calls.append((args, kwargs))
            return passing_audit()

        with patch("blueprint.make_client", return_value=object()), patch(
            "blueprint._call_blueprint_model", return_value=response(baseline_code()),
        ), patch(
            "blueprint.validate_blueprint_fidelity", return_value=[],
        ), patch("blueprint.run_semantic_audit", side_effect=audit):
            generated = generate_blueprint_from_informal(
                "Find the value of the original quantity plus five.",
                "The source COT concludes 45.",
                "target",
                compiler=PassingCompiler(),
                cot_manifest_json=manifest(),
                claimed_answer="45",
                semantic_fidelity_enabled=True,
                semantic_require_step_ids=True,
                semantic_static_gate=True,
                semantic_audit_mode="full",
                max_retries=1,
            )

        self.assertEqual(generated.target_theorem, "target")
        self.assertEqual(len(audit_calls), 1)
        self.assertEqual(
            audit_calls[0][1]["informal_statement"],
            "Find the value of the original quantity plus five.",
        )
        self.assertEqual(audit_calls[0][1]["claimed_answer"], "45")

    def test_phase3_audit_receives_original_problem_and_claimed_answer(self) -> None:
        baseline = _parse_blueprint(baseline_code(), "target")
        audit_calls = []

        def audit(*args, **kwargs):
            audit_calls.append((args, kwargs))
            return passing_audit()

        with patch("refinement.make_client", return_value=object()), patch(
            "refinement._call_blueprint_model", return_value=response(baseline_code()),
        ), patch(
            "refinement.validate_blueprint_fidelity", return_value=[],
        ), patch("refinement.run_semantic_audit", side_effect=audit):
            refined = refine_blueprint(
                baseline,
                OrchestratorResult(root_name="target"),
                PassingCompiler(),
                history=[],
                informal_statement="Find the value of the original quantity plus five.",
                informal_proof="The source COT concludes 45.",
                cot_manifest_json=manifest(),
                claimed_answer="45",
                semantic_fidelity_enabled=True,
                semantic_require_step_ids=True,
                semantic_static_gate=True,
                semantic_freeze_refinement=True,
                semantic_audit_mode="full",
                max_retries=1,
            )

        self.assertEqual(refined.target_theorem, "target")
        self.assertEqual(len(audit_calls), 1)
        self.assertEqual(
            audit_calls[0][1]["informal_statement"],
            "Find the value of the original quantity plus five.",
        )
        self.assertEqual(audit_calls[0][1]["claimed_answer"], "45")

    def test_phase1_static_rejection_precedes_lean(self) -> None:
        bad = """import Mathlib
import Architect
@[blueprint (title := "COT_STEP:S003") (statement := /-- final -/) (proof := /-- source -/)]
theorem target : True := by sorry_using []
"""
        compiler = NeverCompiler()
        with patch("blueprint.make_client", return_value=object()), patch(
            "blueprint._call_blueprint_model", return_value=response(bad),
        ):
            with self.assertRaises(BlueprintGenerationError) as caught:
                generate_blueprint_from_informal(
                    "problem", "cot", "target", compiler=compiler,
                    cot_manifest_json=manifest(), claimed_answer="45",
                    semantic_fidelity_enabled=True,
                    semantic_require_step_ids=True,
                    semantic_static_gate=True,
                    semantic_max_repair_attempts=0,
                    max_retries=1,
                )
        self.assertEqual(caught.exception.failure_stage, "semantic_gate")
        self.assertIn("VACUOUS_TRUE_ROOT", str(caught.exception))
        self.assertEqual(compiler.calls, 0)

    def test_phase3_freeze_rejection_precedes_lean(self) -> None:
        baseline = _parse_blueprint(baseline_code(), "target")
        changed = baseline_code().replace(
            "theorem target : quantity + 5 = 45",
            "theorem target : quantity + 5 = 46",
        )
        compiler = NeverCompiler()
        with patch("refinement.make_client", return_value=object()), patch(
            "refinement._call_blueprint_model", return_value=response(changed),
        ):
            with self.assertRaises(SemanticRefinementError) as caught:
                refine_blueprint(
                    baseline,
                    OrchestratorResult(root_name="target"),
                    compiler,
                    history=[],
                    cot_manifest_json=manifest(),
                    claimed_answer="45",
                    semantic_fidelity_enabled=True,
                    semantic_require_step_ids=True,
                    semantic_static_gate=True,
                    semantic_freeze_refinement=True,
                    semantic_max_repair_attempts=0,
                    max_retries=1,
                )
        self.assertIn("ROOT_SIGNATURE_DRIFT", str(caught.exception))
        self.assertEqual(compiler.calls, 0)

    def test_phase3_retries_keep_only_latest_lean_candidate(self) -> None:
        baseline = _parse_blueprint(baseline_code(), "target")

        def candidate(value: int) -> str:
            return baseline_code().replace("quantity + 5 = 45", f"quantity + 5 = {value}")

        responses = [
            SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content=("PRIVATE_PHASE3_REASONING_ONE\n" * 1000)
                    + f"```lean\n{candidate(46)}\n```",
                    tool_calls=[],
                ),
                finish_reason="length",
            )], usage=None),
            SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content=("PRIVATE_PHASE3_REASONING_TWO\n" * 1000)
                    + f"```lean\n{candidate(47)}\n```",
                    tool_calls=[],
                ),
                finish_reason="stop",
            )], usage=None),
            response(candidate(48)),
        ]
        requests = []

        def call_model(_client, _model, messages, *_args, **_kwargs):
            requests.append(deepcopy(messages))
            return responses[len(requests) - 1]

        compiler = SimpleNamespace(check_blueprint=lambda *_args: CompilerResult(
            False,
            errors=["synthetic refinement error"],
            failure_kind="lean",
        ))
        with patch("refinement.make_client", return_value=object()), patch(
            "refinement._call_blueprint_model", side_effect=call_model,
        ):
            with self.assertRaises(RuntimeError):
                refine_blueprint(
                    baseline,
                    OrchestratorResult(root_name="target"),
                    compiler,
                    history=[],
                    informal_statement="problem",
                    informal_proof="proof",
                    max_retries=3,
                )

        self.assertEqual(requests[0], requests[1][:2])
        self.assertEqual([message["role"] for message in requests[1]], [
            "system", "user", "assistant", "user",
        ])
        self.assertIn("quantity + 5 = 46", requests[1][2]["content"])
        self.assertNotIn("PRIVATE_PHASE3_REASONING_ONE", requests[1][2]["content"])
        self.assertIn("reached its output limit", requests[1][3]["content"])

        self.assertEqual(requests[0], requests[2][:2])
        self.assertIn("quantity + 5 = 47", requests[2][2]["content"])
        self.assertNotIn("quantity + 5 = 46", requests[2][2]["content"])
        self.assertNotIn("PRIVATE_PHASE3_REASONING_TWO", requests[2][2]["content"])


if __name__ == "__main__":
    unittest.main()
