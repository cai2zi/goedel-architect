from __future__ import annotations

import json
import asyncio
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow.parquet as pq
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from blueprint import (  # noqa: E402
    Blueprint,
    BlueprintGenerationError,
    _parse_blueprint,
    generate_blueprint_from_informal,
)
from checkpoint import CheckpointState  # noqa: E402
from cot_blueprint_refine.common import (  # noqa: E402
    claimed_answer,
    extract_boxed_contents,
    extract_post_think,
    load_config,
)
from cot_blueprint_refine.evaluate import (  # noqa: E402
    ANALYSIS_SCHEMA,
    COMPARISON_CSV_FIELDS,
    _judge_decision,
    evaluate,
    grade_final_answer,
    grade_response,
    summarize_ablation,
    summarize_comparisons,
    write_analysis_parquet,
    write_full_analysis_prompt,
    write_pipeline_code_snapshot,
)
from cot_blueprint_refine.judge import (  # noqa: E402
    judge_cache_key,
    judge_equivalences,
    parse_judge_flag,
)
from cot_blueprint_refine.export_blueprint_contexts import (  # noqa: E402
    export_contexts,
    prompt_signal,
    recover_invalid_blueprint,
    revalidate_proved_nodes,
    render_blueprint_context,
)
from cot_blueprint_refine.prepare_inputs import make_generation_row, prepare  # noqa: E402
from cot_blueprint_refine.run_cot_refinement import (  # noqa: E402
    _call_one,
    build_messages,
    conversation_path,
    fit_refinement_messages,
    fit_messages_to_context,
    normalize_refined_output,
    synthesize_legacy_conversation,
)
from cot_blueprint_refine.run_experiment import ExperimentLock, blueprint_results_complete  # noqa: E402
from cot_blueprint_refine.vllm_runtime import (  # noqa: E402
    PersistentVLLMRuntime,
    VLLMServer,
    validate_service_config,
)


class CotCleaningTest(unittest.TestCase):
    def test_extracts_only_suffix_after_balanced_think(self) -> None:
        post, reason = extract_post_think("<think>private</think> Public \\boxed{7}")
        self.assertEqual(reason, "")
        self.assertEqual(post, "Public \\boxed{7}")

    def test_rejects_unclosed_think(self) -> None:
        post, reason = extract_post_think("<think>unfinished")
        self.assertEqual(post, "")
        self.assertEqual(reason, "unclosed_think")

    def test_restores_qwen35_implicit_think_start(self) -> None:
        post, reason = extract_post_think("private reasoning</think>Answer \\boxed{7}")
        self.assertEqual(reason, "")
        self.assertEqual(post, "Answer \\boxed{7}")
        post, reason = extract_post_think("one</think>two</think>Answer \\boxed{7}")
        self.assertEqual(post, "")
        self.assertEqual(reason, "unmatched_think_close")

    def test_nested_box_and_last_box(self) -> None:
        text = r"First \boxed{\frac{1}{2}}, finally \boxed{3}"
        self.assertEqual(extract_boxed_contents(text), [r"\frac{1}{2}", "3"])
        self.assertEqual(claimed_answer(text), "3")

    def test_refined_output_contract(self) -> None:
        output, error, stripped, count = normalize_refined_output(
            "Step.\n\\boxed{4}", "stop"
        )
        self.assertEqual((error, stripped, count), ("", False, 1))
        self.assertEqual(output, "Step.\n\\boxed{4}")
        output, error, _stripped, count = normalize_refined_output(
            "\\boxed{4} and \\boxed{5}", "stop"
        )
        self.assertEqual(output, "")
        self.assertEqual(error, "conflicting_boxed_answers:found_2")
        self.assertEqual(count, 2)

    def test_refined_output_removes_placeholder_and_duplicate_boxes(self) -> None:
        output, error, stripped, count = normalize_refined_output(
            "Instruction `<think>` and \\boxed{...}. Work gives \\boxed{4}. Final \\boxed{4}",
            "stop",
        )
        self.assertEqual(error, "")
        self.assertFalse(stripped)
        self.assertEqual(count, 3)
        self.assertEqual(extract_boxed_contents(output), ["4"])

    def test_refined_output_ignores_analysis_outside_final_markers(self) -> None:
        raw = (
            r"Analysis mentions <think> and \boxed{wrong}. "
            r"<final_refined_solution>Steps. Final \boxed{2}</final_refined_solution>"
        )
        output, error, stripped, count = normalize_refined_output(raw, "stop")
        self.assertEqual(error, "")
        self.assertFalse(stripped)
        self.assertEqual(count, 1)
        self.assertEqual(output, r"Steps. Final \boxed{2}")

    def test_final_answer_math_verify_regressions(self) -> None:
        cases = [
            ("cmimc_2025/38", r"10+\frac{40\pi}{3}", "10", False),
            ("MATH-500/test/intermediate_algebra/1566.json", "2k", "2k", True),
            (
                "MATH-500/test/precalculus/1281.json",
                r"11 \sqrt{5} + 11",
                r"11(1 + \sqrt{5})",
                True,
            ),
            ("brumo_2025/18", r"20\pi", r"20\pi", True),
        ]
        for row_id, gold, candidate, expected in cases:
            with self.subTest(ID=row_id):
                result = grade_final_answer(gold, candidate)
                self.assertEqual(result["is_correct"], expected)
                self.assertEqual(result["scoring_mode"], "canonical_claimed_answer")

        # Whole-COT any_match remains observable, but cannot rescue the wrong
        # final answer for cmimc_2025/38.
        whole_cot = r"An intermediate value is 10. Final: \boxed{10}."
        self.assertTrue(grade_response(r"10+\frac{40\pi}{3}", whole_cot)["is_correct"])
        self.assertFalse(
            grade_final_answer(r"10+\frac{40\pi}{3}", claimed_answer(whole_cot))["is_correct"]
        )


class PrepareInputsTest(unittest.TestCase):
    def test_filters_length_and_does_not_put_gold_in_generation_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "predictions.jsonl"
            rows = [
                {
                    "ID": "ok/1", "source": "demo", "row_index": 0,
                    "problem": "What is 1+1?", "gold": "2", "status": "ok",
                    "finish_reason": "stop",
                    "raw_cot": "<think>reason</think>Answer \\boxed{2}",
                },
                {
                    "ID": "length/open", "source": "demo", "row_index": 1,
                    "problem": "x", "gold": "x", "status": "ok",
                    "finish_reason": "length", "raw_cot": "<think>truncated",
                },
                {
                    "ID": "length/closed", "source": "demo", "row_index": 2,
                    "problem": "x", "gold": "x", "status": "ok",
                    "finish_reason": "length", "raw_cot": "<think>x</think>truncated",
                },
                {
                    "ID": "qwen35/implicit", "source": "demo", "row_index": 3,
                    "problem": "What is 2+2?", "gold": "4", "status": "ok",
                    "finish_reason": "stop",
                    "raw_cot": "reasoning supplied after prompt opener</think>Answer \\boxed{4}",
                },
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            config = OmegaConf.create({
                "input_predictions": str(input_path),
                "output_base": str(root / "outputs"),
                "exp_name": "unit",
                "include_ids": [],
            })
            stats = prepare(config)
            self.assertEqual(stats["unique_rows"], 4)
            self.assertEqual(stats["finish_reason_length"], 2)
            self.assertEqual(stats["length_unclosed_think"], 1)
            self.assertEqual(stats["length_balanced_think"], 1)
            self.assertEqual(stats["implicit_think_start_restored"], 1)
            self.assertEqual(stats["eligible_rows"], 2)
            parquet_path = next((root / "outputs/unit/prepared/data/qwen3_8b_math_verify").glob("*.parquet"))
            parquet_rows = pq.read_table(parquet_path).to_pylist()
            parquet_row = parquet_rows[0]
            self.assertNotIn("gold", parquet_row)
            self.assertEqual(parquet_row["claimed_answer"], "2")
            self.assertEqual(parquet_rows[1]["claimed_answer"], "4")

    def test_generation_statement_contains_claim_not_gold(self) -> None:
        row = {
            "ID": "id", "source": "s", "row_index": 1,
            "problem": "Compute x", "gold": "secret",
        }
        generation = make_generation_row(row, "Proof \\boxed{wrong}", "wrong")
        self.assertIn("Compute x", generation["informal_statement"])
        self.assertIn("\\boxed{wrong}", generation["informal_statement"])
        self.assertNotIn("gold", generation)
        self.assertNotIn("secret", json.dumps(generation))


class BlueprintContextTest(unittest.TestCase):
    @staticmethod
    def _blueprint() -> Blueprint:
        code = """import Mathlib
import Architect

@[blueprint (statement := /-- A checked step. -/) (proof := /-- Trivial. -/)]
lemma checked_step : True := by sorry_using []

@[blueprint (statement := /-- The target. -/) (proof := /-- Uses the step. -/)]
theorem target : True := by sorry_using [checked_step]
"""
        return _parse_blueprint(code, "target")

    def test_solved_proof_is_spliced_and_failed_node_is_annotated(self) -> None:
        blueprint = self._blueprint()
        blueprint.nodes[0].statement = "A checked step.\nWith a second line."
        blueprint.nodes[0].proof_sketch = "Trivial.\nUse True.intro."
        state = CheckpointState(informal_statement="test", model="model")
        state.set_blueprint(blueprint)
        state.proved_cache = {"checked_step": "by trivial"}
        state.node_results = {
            "checked_step": {"signal": "solved", "proof_body": "by trivial", "lean_errors": []},
            "target": {
                "signal": "proof_too_hard",
                "proof_body": "by exact checked_step",
                "lean_errors": ["example error"],
            },
        }
        context, nodes, infra = render_blueprint_context(blueprint, state)
        self.assertFalse(infra)
        self.assertIn("theorem checked_step : True := by trivial", context)
        self.assertIn("COT_BLUEPRINT_NODE_STATEMENT: A checked step.", context)
        self.assertIn("-- With a second line.", context)
        self.assertIn("COT_BLUEPRINT_NODE_PROOF_SKETCH: Trivial.", context)
        self.assertIn("-- Use True.intro.", context)
        self.assertIn("COT_BLUEPRINT_NODE_STATUS: NOT_PROVED", context)
        self.assertNotIn("COT_BLUEPRINT_NODE_NAME", context)
        self.assertNotIn("COT_BLUEPRINT_NODE_KIND", context)
        self.assertIn("theorem target : True := by sorry_using [checked_step]", context)
        self.assertEqual([node["prompt_signal"] for node in nodes], ["PROVED", "NOT_PROVED"])
        self.assertEqual(nodes[0]["statement"], "A checked step.\nWith a second line.")
        self.assertEqual(nodes[0]["proof_sketch"], "Trivial.\nUse True.intro.")

    def test_signal_mapping(self) -> None:
        self.assertEqual(prompt_signal("blocked_by_dependency"), "BLOCKED_BY_DEPENDENCY")
        self.assertEqual(prompt_signal("formally_negated"), "FORMALLY_NEGATED")
        self.assertEqual(prompt_signal("protocol_error"), "NOT_PROVED")
        self.assertEqual(prompt_signal("solved", proved=True), "PROVED")

    def test_stale_proof_becomes_not_proved_and_dependent_is_blocked(self) -> None:
        blueprint = self._blueprint()
        state = CheckpointState(informal_statement="test", model="model")
        state.set_blueprint(blueprint)
        state.proved_cache = {
            "checked_step": "by stale_tactic",
            "target": "by exact checked_step",
        }
        state.node_results = {
            "checked_step": {"signal": "solved", "proof_body": "by stale_tactic"},
            "target": {"signal": "solved", "proof_body": "by exact checked_step"},
        }

        class FakeCompiler:
            def check_node(self, _proof: str, *, node_decl: str, **_kwargs):
                self.last_decl = node_decl
                return SimpleNamespace(
                    success=False,
                    failure_kind="lean",
                    diagnostics=["unsolved goal"],
                )

        overrides, infra = revalidate_proved_nodes(blueprint, state, FakeCompiler())
        self.assertFalse(infra)
        self.assertEqual(overrides["checked_step"]["signal"], "proof_too_hard")
        self.assertEqual(overrides["target"]["signal"], "blocked_by_dependency")
        context, nodes, _ = render_blueprint_context(blueprint, state, overrides)
        self.assertEqual([node["prompt_signal"] for node in nodes], [
            "NOT_PROVED", "BLOCKED_BY_DEPENDENCY",
        ])
        self.assertIn("by sorry_using []", context)
        self.assertIn("proof attempt did not compile", context)
        self.assertIn("unsolved goal", context)
        self.assertNotIn("LAST_SUBMITTED_PROOF", context)


class InvalidBlueprintRecoveryTest(unittest.TestCase):
    def test_saved_artifact_is_preferred_then_trace_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saved = root / "failed.lean"
            saved.write_text("import Mathlib\n@[blueprint] theorem saved : True := by sorry\n")
            trace = root / "trace.jsonl"
            trace.write_text(json.dumps({
                "kind": "llm_response",
                "result": "```lean\nimport Mathlib\n@[blueprint] theorem traced : True := by sorry\n```",
            }) + "\n")
            row = {
                "failed_blueprint_candidate_path": str(saved),
                "trace_path": str(trace),
            }
            candidate, source = recover_invalid_blueprint(row)
            self.assertEqual(source, "saved_artifact")
            self.assertIn("theorem saved", candidate)
            saved.unlink()
            candidate, source = recover_invalid_blueprint(row)
            self.assertEqual(source, "trace_fallback")
            self.assertIn("theorem traced", candidate)

    def test_generation_error_carries_last_invalid_candidate(self) -> None:
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="```lean\nimport Mathlib\ntheorem wrong_name : True := by trivial\n```",
            ),
            finish_reason="stop",
        )])
        with patch("blueprint.make_client", return_value=object()), patch(
            "blueprint._call_blueprint_model", return_value=response,
        ):
            with self.assertRaises(BlueprintGenerationError) as caught:
                generate_blueprint_from_informal(
                    "statement", "proof", "required_name",
                    compiler=SimpleNamespace(), max_retries=1,
                )
        self.assertIn("wrong_name", caught.exception.last_candidate)
        self.assertEqual(caught.exception.failure_stage, "blueprint_contract")


class ContextBudgetTest(unittest.TestCase):
    class CharacterTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return [ord(char) for char in text]

        def decode(self, tokens, skip_special_tokens: bool = True):
            return "".join(chr(token) for token in tokens)

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            text = "".join(message["role"] + message["content"] for message in messages)
            if add_generation_prompt:
                text += "assistant"
            return {"input_ids": self.encode(text), "attention_mask": [1] * len(text)}

    def test_invalid_candidate_is_reference_and_blueprint_is_truncated(self) -> None:
        row = {
            "problem": "Compute 1+1.",
            "claimed_answer": "2",
            "original_cot": "One plus one is two. \\boxed{2}",
            "lean_context": "A" * 12000,
            "context_quality": "INVALID_BLUEPRINT_CANDIDATE",
            "error": "syntax error",
            "nodes": [],
        }
        config = OmegaConf.create({"refine": {
            "tokenizer_path": "unused",
            "context_window": 7000,
            "context_safety_margin": 100,
            "max_tokens": 1000,
        }})
        with patch(
            "cot_blueprint_refine.run_cot_refinement._load_tokenizer",
            return_value=self.CharacterTokenizer(),
        ):
            messages, metadata = fit_messages_to_context(row, config)
        self.assertTrue(metadata["blueprint_truncated"])
        self.assertLessEqual(metadata["input_tokens"] + 1100, 7000)
        self.assertIn("fallible reference", messages[1]["content"])
        self.assertIn("syntax error", messages[1]["content"])

    def test_qwen3_refine_budget_reserves_20480_output_tokens(self) -> None:
        row = {
            "problem": "p", "claimed_answer": "1", "original_cot": r"\boxed{1}",
            "lean_context": "short", "context_quality": "VERIFIED", "nodes": [],
        }
        config = OmegaConf.create({"refine": {
            "tokenizer_path": "unused",
            "context_window": 40960,
            "context_safety_margin": 256,
            "max_tokens": 20480,
        }})
        with patch(
            "cot_blueprint_refine.run_cot_refinement._load_tokenizer",
            return_value=self.CharacterTokenizer(),
        ):
            _messages, metadata = fit_messages_to_context(row, config)
        self.assertEqual(metadata["effective_max_tokens"], 20480)
        self.assertEqual(metadata["max_input_tokens"], 20224)

        base = OmegaConf.load(REPO_ROOT / "experiments/cot_blueprint_refine/configs/base.yaml")
        self.assertEqual(base.refine.max_tokens, 20480)

    def test_cot_only_prompt_has_no_blueprint_data_and_zero_blueprint_tokens(self) -> None:
        row = {
            "problem": "Compute 1+1.",
            "claimed_answer": "2",
            "original_cot": r"The source solution adds the terms. \boxed{2}",
            "lean_context": "SECRET_LEAN_CONTEXT",
            "context_quality": "SECRET_CONTEXT_QUALITY",
            "error": "SECRET_DIAGNOSTIC",
            "nodes": [{"proof_sketch": "SECRET_NODE"}],
        }
        config = OmegaConf.create({"refine": {
            "tokenizer_path": "unused",
            "context_window": 8000,
            "context_safety_margin": 100,
            "max_tokens": 1000,
        }})
        with patch(
            "cot_blueprint_refine.run_cot_refinement._load_tokenizer",
            return_value=self.CharacterTokenizer(),
        ):
            messages, metadata = fit_refinement_messages(
                row,
                config,
                prompt_mode="cot_only",
                source_solution_model_label="Qwen3-8B",
            )
        prompt = "\n".join(message["content"] for message in messages)
        self.assertIn("Qwen3-8B", prompt)
        self.assertIn(row["problem"], prompt)
        self.assertIn(row["original_cot"], prompt)
        for secret in (
            "SECRET_LEAN_CONTEXT", "SECRET_CONTEXT_QUALITY",
            "SECRET_DIAGNOSTIC", "SECRET_NODE",
        ):
            self.assertNotIn(secret, prompt)
        self.assertEqual(metadata["blueprint_tokens_original"], 0)
        self.assertEqual(metadata["blueprint_tokens_used"], 0)
        self.assertFalse(metadata["blueprint_truncated"])


class RefinementConversationTest(unittest.TestCase):
    def test_records_complete_request_response_and_normalization(self) -> None:
        class FakeResponse:
            choices = [SimpleNamespace(
                message=SimpleNamespace(
                    content=r"Checked steps. Final \boxed{2}",
                    reasoning_content="internal reasoning",
                    model_extra={},
                ),
                finish_reason="stop",
            )]

            def model_dump(self, mode="json"):
                return {
                    "id": "response-1",
                    "choices": [{
                        "message": {
                            "content": r"Checked steps. Final \boxed{2}",
                            "reasoning_content": "internal reasoning",
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
                }

        class FakeCompletions:
            async def create(self, **_kwargs):
                return FakeResponse()

        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        with tempfile.TemporaryDirectory() as temporary:
            config = OmegaConf.create({
                "output_base": temporary,
                "exp_name": "unit",
                "refine": {
                    "timeout_s": 30,
                    "max_retries": 2,
                    "retry_max_delay_s": 1,
                    "retry_base_delay_s": 0,
                    "model": "model",
                    "openai_base_url": "http://localhost/v1",
                    "temperature": 0.6,
                    "max_tokens": 100,
                },
            })
            row = {
                "ID": "sample/1", "source": "s", "problem": "1+1?",
                "claimed_answer": "2", "context_quality": "VERIFIED",
                "status": "ready", "refine_eligible": True,
            }
            messages = [{"role": "user", "content": "complete input"}]
            with patch(
                "cot_blueprint_refine.run_cot_refinement.fit_messages_to_context",
                return_value=(messages, {"input_tokens": 10, "effective_max_tokens": 100}),
            ):
                result = asyncio.run(_call_one(client, asyncio.Semaphore(1), row, config))
            self.assertEqual(result["status"], "ok")
            artifact = conversation_path(Path(temporary) / "unit", "sample/1")
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertFalse(payload["reconstructed"])
            self.assertEqual(payload["events"][0]["request"]["messages"], messages)
            self.assertEqual(payload["events"][0]["response"]["id"], "response-1")
            self.assertEqual(payload["events"][0]["assistant_reasoning_content"], "internal reasoning")
            self.assertEqual(payload["events"][0]["normalization"]["status"], "ok")

    def test_legacy_conversation_preserves_existing_prompt_and_response(self) -> None:
        row = {
            "ID": "old", "status": "ok", "attempts": 1,
            "prompt": [{"role": "user", "content": "input"}],
            "raw_response": {"id": "old-response"},
            "raw_content": r"Answer \boxed{1}", "refined_cot": r"Answer \boxed{1}",
        }
        payload = synthesize_legacy_conversation(row)
        event = payload["events"][0]
        self.assertTrue(payload["reconstructed"])
        self.assertEqual(event["request"]["messages"], row["prompt"])
        self.assertEqual(event["response"], row["raw_response"])


class AnalysisArtifactTest(unittest.TestCase):
    def test_writes_typed_parquet_code_snapshot_and_full_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = {field.name: None for field in ANALYSIS_SCHEMA}
            row.update({
                "ID": "id", "source": "s", "before_correct": False,
                "after_correct": True, "transition": "wrong_to_correct",
                "conversation_json": '{"events": []}',
            })
            parquet_path = root / "analysis.parquet"
            write_analysis_parquet(parquet_path, [row])
            table = pq.read_table(parquet_path)
            self.assertEqual(table.num_rows, 1)
            self.assertIn("transition", table.column_names)

            (root / "config_resolved.yaml").write_text("exp_name: unit\n", encoding="utf-8")
            code_path = write_pipeline_code_snapshot(root)
            code = code_path.read_text(encoding="utf-8")
            self.assertIn("experiments/cot_blueprint_refine/run_cot_refinement.py", code)
            self.assertIn("experiments/robustpa_refine/run_robustpa_refine.py", code)
            self.assertIn("config_resolved.yaml", code)

            prompt_path = write_full_analysis_prompt(root, parquet_path, code_path)
            prompt = prompt_path.read_text(encoding="utf-8")
            self.assertIn("conversation_json", prompt)
            self.assertIn("correct→wrong", prompt)
            self.assertIn(parquet_path.name, prompt)


class ExportContextsTest(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _checkpoint(path: Path) -> None:
        blueprint = BlueprintContextTest._blueprint()
        state = CheckpointState(informal_statement="test", model="model", status="solved")
        state.set_blueprint(blueprint)
        state.proved_cache = {"checked_step": "by trivial", "target": "by trivial"}
        state.node_results = {
            "checked_step": {"signal": "solved", "proof_body": "by trivial", "lean_errors": []},
            "target": {"signal": "solved", "proof_body": "by trivial", "lean_errors": []},
        }
        state.save(path)

    def test_concurrent_export_preserves_result_order_and_counts(self) -> None:
        class FakeCompiler:
            def check(self, _lean_context: str, allow_sorry: bool = False):
                return SimpleNamespace(
                    success=True,
                    failure_kind=None,
                    diagnostics=[],
                )

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "out" / "unit"
            prepared = output_root / "prepared" / "generation_inputs.jsonl"
            robustpa = output_root / "robustpa" / "blueprint" / "results.jsonl"
            ckpt_b = root / "checkpoints" / "b.json"
            ckpt_a = root / "checkpoints" / "a.json"
            self._checkpoint(ckpt_b)
            self._checkpoint(ckpt_a)
            self._write_jsonl(prepared, [
                {
                    "name": "b", "source": "s", "problem": "problem b",
                    "claimed_answer": "2", "post_think_cot": "cot b",
                },
                {
                    "name": "a", "source": "s", "problem": "problem a",
                    "claimed_answer": "1", "post_think_cot": "cot a",
                },
            ])
            self._write_jsonl(robustpa, [
                {"source_id": "b", "status": "solved", "root_proved": True, "checkpoint_path": str(ckpt_b)},
                {"source_id": "a", "status": "solved", "root_proved": True, "checkpoint_path": str(ckpt_a)},
            ])
            config = OmegaConf.create({
                "output_base": str(root / "out"),
                "exp_name": "unit",
                "blueprint": {
                    "lean_api_url": "unused",
                    "lean_server_timeout": 1,
                    "lean_max_inflight_snippets": 4,
                    "lean_batch_size": 1,
                },
                "export": {"workers": 2},
            })
            with patch(
                "cot_blueprint_refine.export_blueprint_contexts._make_compiler",
                return_value=FakeCompiler(),
            ):
                metrics = export_contexts(config)
            output_path = Path(metrics["output"])
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["ID"] for row in rows], ["b", "a"])
            self.assertEqual(metrics["counts"]["ready"], 2)
            self.assertEqual(metrics["counts"]["node_PROVED"], 4)
            self.assertIn("COT_BLUEPRINT_NODE_STATEMENT", rows[0]["lean_context"])
            self.assertIn("statement", rows[0]["nodes"][0])
            self.assertIn("proof_sketch", rows[0]["nodes"][0])


class EvaluationMetricsTest(unittest.TestCase):
    def test_comparison_csv_exposes_judge_audit_fields(self) -> None:
        for side in ("before", "after"):
            self.assertIn(f"{side}_extracted_pred", COMPARISON_CSV_FIELDS)
            self.assertIn(f"{side}_math_verify_correct", COMPARISON_CSV_FIELDS)
            self.assertIn(f"{side}_judge_status", COMPARISON_CSV_FIELDS)
            self.assertIn(f"{side}_judge_equivalent", COMPARISON_CSV_FIELDS)
            self.assertIn(f"{side}_judge_reason", COMPARISON_CSV_FIELDS)
            self.assertIn(f"{side}_judge_error", COMPARISON_CSV_FIELDS)
            self.assertIn(f"{side}_correct", COMPARISON_CSV_FIELDS)

    def test_full_denominators_and_transitions(self) -> None:
        comparisons = [
            {
                "source": "s", "before_correct": True, "after_correct": False,
                "transition": "correct_to_wrong", "blueprint_status": "ready",
                "refine_status": "ok", "node_status_counts": {"PROVED": 2},
            },
            {
                "source": "s", "before_correct": False, "after_correct": True,
                "transition": "wrong_to_correct", "blueprint_status": "ready",
                "refine_status": "ok", "node_status_counts": {"NOT_PROVED": 1},
            },
        ]
        metrics = summarize_comparisons(
            comparisons,
            dataset_total=4,
            global_eligible_total=2,
            global_before_correct=1,
            historical_raw_correct=2,
        )
        self.assertEqual(metrics["dataset"]["historical_raw_accuracy"], 0.5)
        self.assertEqual(
            metrics["dataset"]["historical_baseline_scoring"],
            "legacy_whole_cot_math_verify_any_match_diagnostic_only",
        )
        self.assertEqual(
            metrics["dataset"]["current_math_verify_scoring"],
            "canonical_claimed_answer_only",
        )
        self.assertEqual(metrics["dataset"]["strict_post_think_before_full_accuracy"], 0.25)
        self.assertEqual(metrics["full_after"]["eligible_accuracy"], 0.5)
        self.assertEqual(metrics["full_after"]["scoring_method"], "math_verify_or_llm_judge")
        self.assertEqual(metrics["full_after"]["full_accuracy"], 0.25)
        self.assertEqual(metrics["selected"]["node_status_counts"], {"NOT_PROVED": 1, "PROVED": 2})

    def test_judge_assisted_metrics_preserve_math_verify_baseline(self) -> None:
        comparisons = [{
            "source": "s",
            "before_math_verify_correct": False,
            "after_math_verify_correct": False,
            "before_correct": True,
            "after_correct": False,
            "transition": "correct_to_wrong",
            "before_judge_status": "ok",
            "before_judge_equivalent": True,
            "before_judge_cache_hit": True,
            "after_judge_status": "error",
            "after_judge_equivalent": None,
            "blueprint_status": "ready",
            "refine_status": "ok",
            "node_status_counts": {},
        }]
        metrics = summarize_comparisons(
            comparisons,
            dataset_total=2,
            global_eligible_total=1,
            global_before_correct=0,
            historical_raw_correct=0,
        )
        self.assertEqual(metrics["selected"]["math_verify_only"]["before_correct"], 0)
        self.assertEqual(metrics["selected"]["scoring_method"], "math_verify_or_llm_judge")
        self.assertEqual(metrics["selected"]["before_correct"], 1)
        self.assertEqual(metrics["selected"]["judge"]["calls"], 2)
        self.assertEqual(metrics["selected"]["judge"]["errors"], 1)
        self.assertEqual(metrics["selected"]["judge"]["cache_hits"], 1)


class JudgeTest(unittest.TestCase):
    def test_parse_and_cache_key_contract(self) -> None:
        self.assertEqual(
            parse_judge_flag('analysis\n[[JUDGE=1]]'),
            (True, "1"),
        )
        with self.assertRaises(ValueError):
            parse_judge_flag('no decision')
        with self.assertRaises(ValueError):
            parse_judge_flag('[[JUDGE=1]] [[JUDGE=0]]')
        with self.assertRaises(ValueError):
            parse_judge_flag('[[JUDGE=1]] [[JUDGE=1]]')
        first = judge_cache_key(problem="p", gold="1/2", candidate="0.5", model="m")
        same = judge_cache_key(problem="p", gold="1/2", candidate="0.5", model="m")
        changed = judge_cache_key(problem="p", gold="1/2", candidate="0.50", model="m")
        self.assertEqual(first, same)
        self.assertNotEqual(first, changed)

    def test_judge_decision_is_fallback_and_exposes_failures(self) -> None:
        local = _judge_decision(
            math_verify_correct=True, candidate="1", enabled=True, result=None,
        )
        self.assertTrue(local["correct"])
        self.assertEqual(local["status"], "not_needed")
        rescued = _judge_decision(
            math_verify_correct=False,
            candidate="0.5",
            enabled=True,
            result={"status": "ok", "equivalent": True, "reason": "same"},
        )
        self.assertTrue(rescued["correct"])
        failed = _judge_decision(
            math_verify_correct=False,
            candidate="0.5",
            enabled=True,
            result={"status": "error", "error": "timeout"},
        )
        self.assertFalse(failed["correct"])
        self.assertEqual(failed["error"], "timeout")
        unavailable = _judge_decision(
            math_verify_correct=False, candidate="", enabled=True, result=None,
        )
        self.assertEqual(unavailable["status"], "unavailable")

    def test_successful_judgment_is_cached_by_content(self) -> None:
        class FakeResponse:
            choices = [SimpleNamespace(
                message=SimpleNamespace(
                    content='[[JUDGE=1]]',
                    reasoning_content="checked",
                    model_extra={},
                ),
                finish_reason="stop",
            )]

            def model_dump(self, mode="json"):
                return {"id": "judge-1", "choices": []}

        class FakeCompletions:
            calls = 0
            last_kwargs = None

            async def create(self, **kwargs):
                self.calls += 1
                self.last_kwargs = kwargs
                return FakeResponse()

        class FakeClient:
            completions = FakeCompletions()

            def __init__(self, **_kwargs):
                self.chat = SimpleNamespace(completions=self.completions)

            async def close(self):
                pass

        config = OmegaConf.create({
            "resume": True,
            "judge": {
                "model": "judge-model", "api_key": "dummy",
                "openai_base_url": "http://localhost:8001/v1",
                "temperature": 0, "max_tokens": 32, "timeout_s": 2,
                "max_retries": 1, "retry_base_delay_s": 0,
                "retry_max_delay_s": 0, "concurrency": 2,
            },
        })
        request = [
            {
                "ID": "one", "side": "after:blueprint", "variant": "blueprint",
                "problem": "half?", "gold": "1/2", "candidate": "0.5",
            },
            {
                "ID": "one", "side": "after:cot_only", "variant": "cot_only",
                "problem": "half?", "gold": "1/2", "candidate": "0.5",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary, patch(
            "cot_blueprint_refine.judge.AsyncOpenAI", FakeClient,
        ):
            path = Path(temporary) / "judge.jsonl"
            first = asyncio.run(judge_equivalences(request, config, path))
            second = asyncio.run(judge_equivalences(request, config, path))
        self.assertTrue(first[("one", "after:blueprint")]["equivalent"])
        self.assertFalse(first[("one", "after:blueprint")]["cache_hit"])
        self.assertEqual(first[("one", "after:blueprint")]["variant"], "blueprint")
        self.assertEqual(first[("one", "after:cot_only")]["variant"], "cot_only")
        self.assertTrue(second[("one", "after:blueprint")]["cache_hit"])
        self.assertTrue(second[("one", "after:cot_only")]["cache_hit"])
        self.assertEqual(FakeClient.completions.calls, 1)
        self.assertNotIn("response_format", FakeClient.completions.last_kwargs)
        self.assertEqual(FakeClient.completions.last_kwargs["max_tokens"], 32)
        self.assertEqual(
            FakeClient.completions.last_kwargs["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_judge_retries_transient_failures(self) -> None:
        class FakeResponse:
            choices = [SimpleNamespace(
                message=SimpleNamespace(
                    content='[[JUDGE=0]]',
                    reasoning_content=None,
                    model_extra={},
                ),
                finish_reason="stop",
            )]

            def model_dump(self, mode="json"):
                return {"id": "judge-retry", "choices": []}

        class FlakyCompletions:
            calls = 0

            async def create(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError("temporary timeout")
                return FakeResponse()

        class FakeClient:
            completions = FlakyCompletions()

            def __init__(self, **_kwargs):
                self.chat = SimpleNamespace(completions=self.completions)

            async def close(self):
                pass

        config = OmegaConf.create({
            "resume": False,
            "judge": {
                "model": "judge-model", "api_key": "dummy",
                "openai_base_url": "http://localhost:8001/v1",
                "temperature": 0, "max_tokens": 32, "timeout_s": 2,
                "max_retries": 2, "retry_base_delay_s": 0,
                "retry_max_delay_s": 0, "concurrency": 1,
            },
        })
        request = [{
            "ID": "one", "side": "after", "problem": "one?",
            "gold": "1", "candidate": "2",
        }]
        with tempfile.TemporaryDirectory() as temporary, patch(
            "cot_blueprint_refine.judge.AsyncOpenAI", FakeClient,
        ):
            result = asyncio.run(
                judge_equivalences(request, config, Path(temporary) / "judge.jsonl")
            )[("one", "after")]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(FakeClient.completions.calls, 2)
        self.assertEqual([row["status"] for row in result["attempt_log"]], ["error", "ok"])
        self.assertEqual(result["attempt_log"][0]["error_layer"], "api_request")
        self.assertIsNotNone(result["attempt_log"][1]["raw_body"])
        self.assertIsNotNone(result["attempt_log"][1]["raw_content"])

    def test_judge_separates_api_and_flag_decoding_layers(self) -> None:
        class RawHTTPResponse:
            text = '{"malformed transport"'
            request_id = "req-transport"
            status_code = 200

            def parse(self):
                raise json.JSONDecodeError("transport body", self.text, 1)

        class RawCreate:
            async def create(self, **_kwargs):
                return RawHTTPResponse()

        class TransportCompletions:
            with_raw_response = SimpleNamespace(create=RawCreate().create)

        class TransportClient:
            completions = TransportCompletions()

            def __init__(self, **_kwargs):
                self.chat = SimpleNamespace(completions=self.completions)

            async def close(self):
                pass

        class ContentResponse:
            _request_id = "req-content"
            choices = [SimpleNamespace(
                message=SimpleNamespace(
                    content='no judge flag', reasoning_content="reasoning", model_extra={},
                ),
                finish_reason="stop",
            )]

            def model_dump(self, mode="json"):
                return {"id": "content-response", "choices": []}

        class ContentCompletions:
            async def create(self, **_kwargs):
                return ContentResponse()

        class ContentClient:
            completions = ContentCompletions()

            def __init__(self, **_kwargs):
                self.chat = SimpleNamespace(completions=self.completions)

            async def close(self):
                pass

        config = OmegaConf.create({
            "resume": False,
            "judge": {
                "model": "judge-model", "api_key": "dummy",
                "openai_base_url": "http://localhost:8001/v1",
                "temperature": 0, "max_tokens": 32, "timeout_s": 2,
                "max_retries": 1, "retry_base_delay_s": 0,
                "retry_max_delay_s": 0, "concurrency": 1,
            },
        })
        request = [{
            "ID": "one", "side": "before", "problem": "p", "gold": "1", "candidate": "2",
        }]
        with tempfile.TemporaryDirectory() as temporary, patch(
            "cot_blueprint_refine.judge.AsyncOpenAI", TransportClient,
        ):
            transport = asyncio.run(judge_equivalences(
                request, config, Path(temporary) / "transport.jsonl",
            ))[("one", "before")]
        self.assertEqual(transport["error_layer"], "api_response_decoding")
        self.assertEqual(transport["request_id"], "req-transport")
        self.assertEqual(transport["http_status"], 200)
        self.assertEqual(transport["raw_body"], RawHTTPResponse.text)
        self.assertIsNone(transport["raw_content"])

        with tempfile.TemporaryDirectory() as temporary, patch(
            "cot_blueprint_refine.judge.AsyncOpenAI", ContentClient,
        ):
            content = asyncio.run(judge_equivalences(
                request, config, Path(temporary) / "content.jsonl",
            ))[("one", "before")]
        self.assertEqual(content["error_layer"], "judge_flag_missing")
        self.assertEqual(content["request_id"], "req-content")
        self.assertEqual(content["raw_content"], "no judge flag")
        self.assertIsNotNone(content["raw_body"])

    def test_judge_resume_retries_only_failed_cache_key(self) -> None:
        class FakeResponse:
            choices = [SimpleNamespace(
                message=SimpleNamespace(
                    content='[[JUDGE=0]]',
                    reasoning_content=None,
                    model_extra={},
                ),
                finish_reason="stop",
            )]

            def model_dump(self, mode="json"):
                return {"id": "new-response", "choices": []}

        class CountingCompletions:
            calls = 0

            async def create(self, **_kwargs):
                self.calls += 1
                return FakeResponse()

        class FakeClient:
            completions = CountingCompletions()

            def __init__(self, **_kwargs):
                self.chat = SimpleNamespace(completions=self.completions)

            async def close(self):
                pass

        config = OmegaConf.create({
            "resume": True,
            "judge": {
                "model": "judge-model", "api_key": "dummy",
                "openai_base_url": "http://localhost:8001/v1",
                "temperature": 0, "max_tokens": 32, "timeout_s": 2,
                "max_retries": 1, "retry_base_delay_s": 0,
                "retry_max_delay_s": 0, "concurrency": 1,
            },
        })
        requests = [
            {"ID": "cached", "side": "before", "problem": "p1", "gold": "1", "candidate": "1"},
            {"ID": "failed", "side": "after", "problem": "p2", "gold": "1", "candidate": "2"},
        ]
        cached_key = judge_cache_key(
            problem="p1", gold="1", candidate="1", model="judge-model",
        )
        failed_key = judge_cache_key(
            problem="p2", gold="1", candidate="2", model="judge-model",
        )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "cot_blueprint_refine.judge.AsyncOpenAI", FakeClient,
        ):
            path = Path(temporary) / "judge.jsonl"
            path.write_text(
                json.dumps({"cache_key": cached_key, "status": "ok", "equivalent": True})
                + "\n"
                + json.dumps({"cache_key": failed_key, "status": "error", "error": "old"})
                + "\n",
                encoding="utf-8",
            )
            results = asyncio.run(judge_equivalences(requests, config, path))
        self.assertTrue(results[("cached", "before")]["cache_hit"])
        self.assertFalse(results[("failed", "after")]["cache_hit"])
        self.assertEqual(FakeClient.completions.calls, 1)

    def test_failed_judgments_are_retried_on_resume(self) -> None:
        class FailingCompletions:
            calls = 0

            async def create(self, **_kwargs):
                self.calls += 1
                raise RuntimeError("judge unavailable")

        class FakeClient:
            completions = FailingCompletions()

            def __init__(self, **_kwargs):
                self.chat = SimpleNamespace(completions=self.completions)

            async def close(self):
                pass

        config = OmegaConf.create({
            "resume": True,
            "judge": {
                "model": "judge-model", "api_key": "dummy",
                "openai_base_url": "http://localhost:8001/v1",
                "temperature": 0, "max_tokens": 32, "timeout_s": 2,
                "max_retries": 1, "retry_base_delay_s": 0,
                "retry_max_delay_s": 0, "concurrency": 1,
            },
        })
        request = [{
            "ID": "one", "side": "before", "problem": "one?",
            "gold": "1", "candidate": "2",
        }]
        with tempfile.TemporaryDirectory() as temporary, patch(
            "cot_blueprint_refine.judge.AsyncOpenAI", FakeClient,
        ):
            path = Path(temporary) / "judge.jsonl"
            first = asyncio.run(judge_equivalences(request, config, path))
            second = asyncio.run(judge_equivalences(request, config, path))
            saved_rows = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(first[("one", "before")]["status"], "error")
        self.assertEqual(second[("one", "before")]["status"], "error")
        self.assertEqual(FakeClient.completions.calls, 2)
        self.assertEqual(len(saved_rows), 2)


class AblationConfigurationTest(unittest.TestCase):
    def test_qwen8b_profile_uses_one_397b_service_and_two_arms(self) -> None:
        config = load_config("qwen3_8b_397b_refine_ablation", [])
        self.assertTrue(config.resume)
        self.assertEqual(config.refine.source_solution_model_label, "Qwen3-8B")
        self.assertEqual(set(config.refine.variants), {"blueprint", "cot_only"})
        self.assertEqual(config.refine.variants.blueprint.prompt_mode, "blueprint")
        self.assertEqual(config.refine.variants.cot_only.prompt_mode, "cot_only")
        for stage in (config.blueprint, config.refine, config.judge):
            self.assertEqual(stage.model, "Qwen3.5-397B-A17B-FP8")
            self.assertEqual(stage.openai_base_url, "http://127.0.0.1:8001/v1")
            self.assertIsNone(stage.vllm.get("reasoning_parser"))
            self.assertEqual(stage.vllm.tool_call_parser, "qwen3_xml")
            validate_service_config(str(stage.model), str(stage.openai_base_url), stage.vllm)
        self.assertEqual(config.blueprint.phase2_node_concurrency, 512)
        self.assertEqual(config.blueprint.lean_max_inflight_snippets, 48)
        self.assertEqual(config.blueprint.lean_batch_size, 8)
        self.assertEqual(config.blueprint.lean_parallel_batches, 6)
        self.assertTrue(config.blueprint.lean_global_batching)
        self.assertEqual(config.judge.max_tokens, 32)
        self.assertEqual(config.judge.max_retries, 2)
        self.assertEqual(
            config.refine.tokenizer_path,
            "/ssd/czx/models/Qwen3.5-397B-A17B-FP8",
        )

    def test_paired_metrics_keep_errors_in_primary_denominator(self) -> None:
        paired = [
            {
                "ID": "both", "original_correct": False,
                "original_math_verify_correct": False,
                "blueprint_correct": True, "blueprint_math_verify_correct": True,
                "cot_only_correct": True, "cot_only_math_verify_correct": True,
                "blueprint_refine_status": "ok", "cot_only_refine_status": "ok",
                "paired_outcome": "both_correct",
            },
            {
                "ID": "bp-error", "original_correct": True,
                "original_math_verify_correct": True,
                "blueprint_correct": False, "blueprint_math_verify_correct": False,
                "cot_only_correct": True, "cot_only_math_verify_correct": True,
                "blueprint_refine_status": "error", "cot_only_refine_status": "ok",
                "paired_outcome": "cot_only_only_correct",
            },
        ]
        comparison_stub = {
            "blueprint": [
                {"ID": "both", "refine_status": "ok", "transition": "wrong_to_correct"},
                {"ID": "bp-error", "refine_status": "error", "transition": "correct_to_wrong"},
            ],
            "cot_only": [
                {"ID": "both", "refine_status": "ok", "transition": "wrong_to_correct"},
                {"ID": "bp-error", "refine_status": "ok", "transition": "correct_to_correct"},
            ],
        }
        metrics = summarize_ablation(paired, comparison_stub)
        self.assertEqual(metrics["total"], 2)
        self.assertEqual(metrics["arms"]["blueprint"]["math_verify_or_llm_judge"]["correct"], 1)
        self.assertEqual(metrics["paired_outcome_counts"]["cot_only_only_correct"], 1)
        self.assertEqual(metrics["both_generated_ok_diagnostic"]["total"], 1)

    def test_two_sample_ablation_smoke_writes_all_three_answer_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            predictions.write_text("".join(
                json.dumps(row) + "\n" for row in (
                    {
                        "ID": "a", "source": "unit", "status": "ok",
                        "finish_reason": "stop", "problem": "What is one?", "gold": "1",
                        "raw_cot": r"<think>hidden-a</think>Draft. \boxed{1}",
                    },
                    {
                        "ID": "b", "source": "unit", "status": "ok",
                        "finish_reason": "stop", "problem": "What is two?", "gold": "2",
                        "raw_cot": r"<think>hidden-b</think>Draft. \boxed{3}",
                    },
                )
            ), encoding="utf-8")
            output = root / "out" / "unit"

            def write_rows(path: Path, rows: list[dict]) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )

            write_rows(output / "prepared/generation_inputs.jsonl", [
                {"name": "a", "post_think_cot": r"Draft. \boxed{1}", "claimed_answer": "1"},
                {"name": "b", "post_think_cot": r"Draft. \boxed{3}", "claimed_answer": "3"},
            ])
            write_rows(output / "blueprint_contexts/blueprint_contexts.jsonl", [
                {"ID": row_id, "status": "ready", "context_quality": "VERIFIED", "nodes": []}
                for row_id in ("a", "b")
            ])
            for variant, answers in (
                ("blueprint", {"a": "1", "b": "2"}),
                ("cot_only", {"a": "4", "b": "2"}),
            ):
                write_rows(output / f"refinement/{variant}/refined_predictions.jsonl", [
                    {
                        "ID": row_id, "status": "ok", "refine_variant": variant,
                        "prompt_mode": variant, "blueprint_used": variant == "blueprint",
                        "refined_cot": rf"Refined. \boxed{{{answer}}}", "attempts": 1,
                    }
                    for row_id, answer in answers.items()
                ])
            config = OmegaConf.create({
                "input_predictions": str(predictions),
                "output_base": str(root / "out"),
                "exp_name": "unit",
                "resume": False,
                "refine": {
                    "model": "397b", "openai_base_url": "http://localhost/v1",
                    "temperature": 0.6, "max_tokens": 100, "timeout_s": 10,
                    "source_solution_model_label": "Qwen3-8B",
                    "variants": {
                        "blueprint": {"enabled": True, "prompt_mode": "blueprint"},
                        "cot_only": {"enabled": True, "prompt_mode": "cot_only"},
                    },
                },
                "judge": {"enabled": False},
            })
            metrics = evaluate(config)
            self.assertEqual(metrics["ablation"]["total"], 2)
            paired = [
                json.loads(line) for line in
                (output / "evaluation/ablation/paired_comparison.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {"original_answer", "blueprint_answer", "cot_only_answer"} - set(paired[0]),
                set(),
            )
            self.assertTrue((output / "evaluation/blueprint/metrics.json").exists())
            self.assertTrue((output / "evaluation/cot_only/metrics.json").exists())
            self.assertEqual(metrics["ablation"]["paired_outcome_counts"], {
                "blueprint_only_correct": 1,
                "both_correct": 1,
            })


class PersistentVLLMRuntimeTest(unittest.TestCase):
    def test_identical_stages_start_once_reuse_pid_and_stop_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = VLLMServerTest._config(Path(temporary))

            def fake_start(server):
                server.started_at = "start"
                server.process = SimpleNamespace(pid=397, returncode=None)

            def fake_stop(server, **_kwargs):
                server.process.returncode = 0

            with patch.object(VLLMServer, "start", autospec=True, side_effect=fake_start) as start, patch.object(
                VLLMServer, "stop", autospec=True, side_effect=fake_stop,
            ) as stop:
                runtime = PersistentVLLMRuntime(config)
                for stage in ("blueprint", "export", "refine/blueprint", "refine/cot_only", "evaluate/judge"):
                    runtime.ensure(
                        stage=stage,
                        client_model="model",
                        base_url="http://127.0.0.1:8123/v1",
                        service=config.service,
                    )
                self.assertEqual(runtime.pid, 397)
                runtime.close()
            self.assertEqual(start.call_count, 1)
            self.assertEqual(stop.call_count, 1)
            session = json.loads(runtime.session_path.read_text(encoding="utf-8"))
            self.assertEqual(session["start_count"], 1)
            self.assertEqual(session["stop_count"], 1)
            self.assertEqual(session["reuse_count"], 4)
            attachments = [
                json.loads(line)
                for line in runtime.attachments_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual({row["pid"] for row in attachments}, {397})


class VLLMServerTest(unittest.TestCase):
    @staticmethod
    def _config(root: Path, *, auto_start: bool = True, auto_destroy: bool = True):
        model = root / "model"
        model.mkdir(exist_ok=True)
        return OmegaConf.create({
            "output_base": str(root / "outputs"),
            "exp_name": "unit",
            "python_bin": "/env/bin/python",
            "vllm": {
                "auto_start": auto_start,
                "auto_destroy": auto_destroy,
                "startup_timeout_s": 1,
                "shutdown_timeout_s": 1,
                "poll_interval_s": 0,
                "cuda_visible_devices": "0,1",
            },
            "service": {
                "model_path": str(model),
                "served_model_name": "model",
                "host": "127.0.0.1",
                "port": 8123,
                "tensor_parallel_size": 2,
                "max_model_len": 4096,
                "max_num_seqs": 8,
                "gpu_memory_utilization": 0.5,
                "trust_remote_code": True,
                "reasoning_parser": "qwen3",
                "tool_call_parser": "qwen3_coder",
                "enable_auto_tool_choice": True,
                "extra_args": ["--disable-log-stats"],
            },
        })

    def test_validates_client_endpoint_and_builds_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            validate_service_config("model", "http://localhost:8123/v1", config.service)
            with self.assertRaisesRegex(ValueError, "port"):
                validate_service_config("model", "http://localhost:9999/v1", config.service)
            server = VLLMServer(
                config, stage="refine", client_model="model",
                base_url="http://127.0.0.1:8123/v1", service=config.service,
            )
            command = server.command()
            self.assertEqual(command[:3], ["/env/bin/python", "-m", "vllm.entrypoints.openai.api_server"])
            self.assertEqual(command[3:5], ["--model", str(Path(temporary) / "model")])
            self.assertIn("--tensor-parallel-size", command)
            self.assertIn("--tool-call-parser", command)
            self.assertIn("--enable-auto-tool-choice", command)
            self.assertIn("--disable-log-stats", command)

    def test_exclusive_port_rejects_without_starting_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            server = VLLMServer(
                config, stage="blueprint", client_model="model",
                base_url="http://127.0.0.1:8123/v1", service=config.service,
            )
            with patch.object(server, "_port_is_in_use", return_value=True), patch(
                "cot_blueprint_refine.vllm_runtime.subprocess.Popen",
            ) as popen:
                with self.assertRaisesRegex(RuntimeError, "exclusive port"):
                    server.start()
            popen.assert_not_called()

    def test_ready_poll_waits_for_matching_served_model(self) -> None:
        class RunningProcess:
            pid = 4000

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            server = VLLMServer(
                config, stage="blueprint", client_model="model",
                base_url="http://127.0.0.1:8123/v1", service=config.service,
            )
            server.process = RunningProcess()
            with patch.object(
                server, "_available_models", side_effect=[set(), {"model"}],
            ) as available, patch("cot_blueprint_refine.vllm_runtime.time.sleep"):
                server._wait_until_ready()
            self.assertEqual(available.call_count, 2)
            self.assertIsNotNone(server.ready_at)

    def test_exited_startup_reports_log_tail(self) -> None:
        class ExitedProcess:
            pid = 4001
            returncode = 2

            def poll(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            server = VLLMServer(
                config, stage="blueprint", client_model="model",
                base_url="http://127.0.0.1:8123/v1", service=config.service,
            )
            server.root.mkdir(parents=True, exist_ok=True)
            server.log_path.write_text("fatal startup detail\n", encoding="utf-8")
            server.process = ExitedProcess()
            with self.assertRaisesRegex(RuntimeError, "fatal startup detail"):
                server._wait_until_ready()

    def test_startup_health_failure_cleans_process_group(self) -> None:
        class FakeProcess:
            pid = 4100
            returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = 0
                return 0

        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            server = VLLMServer(
                config, stage="blueprint", client_model="model",
                base_url="http://127.0.0.1:8123/v1", service=config.service,
            )
            with patch.object(server, "_port_is_in_use", return_value=False), patch.object(
                server, "_wait_until_ready", side_effect=TimeoutError("not ready"),
            ), patch(
                "cot_blueprint_refine.vllm_runtime.subprocess.Popen",
                return_value=FakeProcess(),
            ), patch(
                "cot_blueprint_refine.vllm_runtime.os.getpgid", return_value=4100,
            ), patch("cot_blueprint_refine.vllm_runtime.os.killpg") as killpg:
                with self.assertRaisesRegex(TimeoutError, "not ready"):
                    server.start()
            killpg.assert_called_once_with(4100, signal.SIGTERM)
            metadata = json.loads(server.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "startup_failed")

    def test_owned_process_group_is_stopped(self) -> None:
        class FakeProcess:
            pid = 4321
            returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = 0
                return 0

        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            server = VLLMServer(
                config, stage="evaluate", client_model="model",
                base_url="http://127.0.0.1:8123/v1", service=config.service,
            )
            fake = FakeProcess()
            with patch.object(server, "_port_is_in_use", return_value=False), patch.object(
                server, "_wait_until_ready", side_effect=lambda: setattr(server, "ready_at", "now"),
            ), patch(
                "cot_blueprint_refine.vllm_runtime.subprocess.Popen", return_value=fake,
            ) as popen, patch(
                "cot_blueprint_refine.vllm_runtime.os.getpgid", return_value=4321,
            ), patch("cot_blueprint_refine.vllm_runtime.os.killpg") as killpg:
                with server:
                    self.assertIs(server.process, fake)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertTrue(popen.call_args.kwargs["env"]["PATH"].startswith("/env/bin:"))
            killpg.assert_called_once()
            self.assertEqual(killpg.call_args.args[0], 4321)
            metadata = json.loads(server.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "stopped")

    def test_exception_and_interrupt_stop_owned_process_group(self) -> None:
        class FakeProcess:
            pid = 4500
            returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = 0
                return 0

        for exception_type in (RuntimeError, KeyboardInterrupt):
            with self.subTest(exception_type=exception_type), tempfile.TemporaryDirectory() as temporary:
                config = self._config(Path(temporary))
                server = VLLMServer(
                    config, stage="refine", client_model="model",
                    base_url="http://127.0.0.1:8123/v1", service=config.service,
                )
                with patch.object(server, "_port_is_in_use", return_value=False), patch.object(
                    server, "_wait_until_ready",
                ), patch(
                    "cot_blueprint_refine.vllm_runtime.subprocess.Popen",
                    return_value=FakeProcess(),
                ), patch(
                    "cot_blueprint_refine.vllm_runtime.os.getpgid", return_value=4500,
                ), patch("cot_blueprint_refine.vllm_runtime.os.killpg") as killpg:
                    with self.assertRaises(exception_type):
                        with server:
                            raise exception_type("stage failed")
                killpg.assert_called_once_with(4500, signal.SIGTERM)
                metadata = json.loads(server.metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    metadata["stop_reason"], f"stage_exception:{exception_type.__name__}"
                )

    def test_external_mode_never_starts_or_kills(self) -> None:
        for auto_destroy in (False, True):
            with self.subTest(auto_destroy=auto_destroy), tempfile.TemporaryDirectory() as temporary:
                config = self._config(
                    Path(temporary), auto_start=False, auto_destroy=auto_destroy,
                )
                server = VLLMServer(
                    config, stage="refine", client_model="model",
                    base_url="http://127.0.0.1:8123/v1", service=config.service,
                )
                with patch(
                    "cot_blueprint_refine.vllm_runtime.subprocess.Popen",
                ) as popen, patch(
                    "cot_blueprint_refine.vllm_runtime.os.killpg",
                ) as killpg:
                    with server:
                        pass
                popen.assert_not_called()
                killpg.assert_not_called()

    def test_auto_destroy_false_leaves_owned_process_running(self) -> None:
        class FakeProcess:
            pid = 5000
            returncode = None

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary), auto_destroy=False)
            server = VLLMServer(
                config, stage="refine", client_model="model",
                base_url="http://127.0.0.1:8123/v1", service=config.service,
            )
            with patch.object(server, "_port_is_in_use", return_value=False), patch.object(
                server, "_wait_until_ready",
            ), patch(
                "cot_blueprint_refine.vllm_runtime.subprocess.Popen",
                return_value=FakeProcess(),
            ), patch("cot_blueprint_refine.vllm_runtime.os.killpg") as killpg:
                with server:
                    pass
            killpg.assert_not_called()
            metadata = json.loads(server.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "left_running")

    def test_shutdown_timeout_force_kills_process_group(self) -> None:
        class SlowProcess:
            pid = 6000
            returncode = None
            waits = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.waits += 1
                if self.waits == 1:
                    raise subprocess.TimeoutExpired("vllm", timeout)
                self.returncode = -9
                return self.returncode

        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            server = VLLMServer(
                config, stage="evaluate", client_model="model",
                base_url="http://127.0.0.1:8123/v1", service=config.service,
            )
            server.process = SlowProcess()
            with patch(
                "cot_blueprint_refine.vllm_runtime.os.getpgid", return_value=6000,
            ), patch("cot_blueprint_refine.vllm_runtime.os.killpg") as killpg:
                server.stop()
            self.assertEqual(
                [call.args[1] for call in killpg.call_args_list],
                [signal.SIGTERM, signal.SIGKILL],
            )
            self.assertTrue(server._forced_kill)


class ExperimentLockTest(unittest.TestCase):
    def test_rejects_concurrent_writer_for_same_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ExperimentLock(root):
                with self.assertRaisesRegex(RuntimeError, "already locked"):
                    with ExperimentLock(root):
                        pass


class BlueprintResumeTest(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_error_results_are_terminal_when_retry_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "out" / "unit"
            self._write_jsonl(output_root / "prepared" / "generation_inputs.jsonl", [
                {"name": "ok"},
                {"name": "err"},
            ])
            self._write_jsonl(output_root / "robustpa" / "blueprint" / "results.jsonl", [
                {"source_id": "ok", "status": "exhausted", "root_proved": False},
                {"source_id": "err", "status": "error", "root_proved": False},
            ])
            config = OmegaConf.create({
                "output_base": str(root / "out"),
                "exp_name": "unit",
                "resume": True,
                "blueprint": {"retry_error_results": False},
            })
            self.assertTrue(blueprint_results_complete(config))
            config.blueprint.retry_error_results = True
            self.assertFalse(blueprint_results_complete(config))

    def test_run_all_uses_stage_environment_variable(self) -> None:
        script = (REPO_ROOT / "experiments/cot_blueprint_refine/run_all.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('STAGE="${STAGE:-all}"', script)
        self.assertIn('--stage "${STAGE}"', script)


if __name__ == "__main__":
    unittest.main()
