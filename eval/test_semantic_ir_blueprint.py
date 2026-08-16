from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

from experiments.semantic_ir_blueprint.conversation import capture_chat_once
from experiments.semantic_ir_blueprint.run_experiment import run_record
from experiments.semantic_ir_blueprint.semantic_ir import SemanticIR
from experiments.semantic_ir_blueprint.source_units import (
    CLOSE_MARKER,
    OPEN_MARKER,
    SourceUnitError,
    make_boundary_anchors,
    parse_boundaries,
    source_units_from_boundaries,
)
from kimina_lean_compiler import CompilerResult


def valid_ir() -> dict:
    return {
        "definitions": [{
            "id": "sample_space",
            "params": [{"name": "k", "type": "positive real number"}],
            "type": "set of points",
            "definition": "the interior of the rectangle with aspect ratio k",
            "source_units": ["S001"],
            "source_description": "The point is sampled from the rectangle interior.",
        }],
        "nodes": [
            {
                "id": "n_relation",
                "kind": "lemma",
                "depends_on": [],
                "claim": {
                    "form": "relation",
                    "binders": [],
                    "assumptions": [],
                    "lhs": "target_region(k)",
                    "relation": "is congruent to modulo seven",
                    "rhs": "diamond_region(k)",
                },
                "source_units": ["S001"],
                "source_description": "The regions are related as stated.",
            },
            {
                "id": "n_predicate",
                "kind": "lemma",
                "depends_on": ["n_relation"],
                "claim": {
                    "form": "predicate",
                    "binders": [],
                    "assumptions": [],
                    "predicate": "is_a_rhombus",
                    "arguments": ["diamond_region(k)"],
                },
                "source_units": ["S001"],
                "source_description": "The region is asserted to be a rhombus.",
            },
            {
                "id": "n_final",
                "kind": "theorem",
                "depends_on": ["n_predicate"],
                "claim": {
                    "form": "proposition",
                    "binders": [{"name": "k", "type": "positive real number"}],
                    "assumptions": ["the rectangle has aspect ratio k"],
                    "proposition": "the target probability equals the claimed expression",
                },
                "source_units": ["S001"],
                "source_description": "The COT concludes the claimed probability.",
            },
        ],
    }


class FakeResponse:
    def __init__(
        self,
        content: str,
        reasoning: str | None = "full hidden reasoning",
        *,
        reasoning_field: str = "reasoning_content",
    ):
        message_values = {
            "content": content,
            "reasoning_content": None,
            "reasoning": None,
            "model_extra": {reasoning_field: reasoning},
        }
        message_values[reasoning_field] = reasoning
        message = SimpleNamespace(**message_values)
        self.choices = [SimpleNamespace(message=message, finish_reason="stop")]
        self.usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=20, total_tokens=30,
        )
        self._content = content
        self._reasoning = reasoning

    def model_dump(self, mode: str = "python") -> dict:
        del mode
        return {
            "id": "response-id",
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": self._content,
                    "reasoning_content": self._reasoning,
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeCompiler:
    def __init__(self, result: CompilerResult | Exception):
        self.result = result
        self.calls = []

    def check_blueprint(self, code: str, target: str):
        self.calls.append((code, target))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def config() -> dict:
    return {
        "experiment_name": "test",
        "model": "fake-model",
        "target_theorem": "n_final",
        "tokenizer_path": "/unused",
        "model_max_context": 40960,
        "context_safety_margin": 512,
        "source_split": {
            "enable_thinking": True, "temperature": 0.0,
            "max_completion_tokens": 2048,
        },
        "semantic_ir": {
            "enable_thinking": True, "temperature": 0.6,
            "top_p": 0.95, "top_k": 20, "max_completion_tokens": 16384,
        },
        "blueprint": {
            "enable_thinking": True, "temperature": 0.6,
            "top_p": 0.95, "top_k": 20,
        },
    }


RECORD = {
    "name": "MATH-500/test/counting_and_probability/731.json",
    "source": "MATH-500",
    "row_index": 444,
    "problem": "A probability problem.",
    "claimed_answer": "1/2",
    "informal_proof": "First claim. Second claim.",
}


LEAN = """```lean
import Mathlib
import Architect
import GoedelArch

@[blueprint] def sample_space (k : ℝ) : Set ℝ := Set.univ
@[blueprint] lemma n_relation : 1 = 1 := by sorry_using []
@[blueprint] lemma n_predicate : 1 = 1 := by sorry_using [n_relation]
@[blueprint] theorem n_final : 1 = 1 := by sorry_using [n_predicate]
```"""


def budgeter(*args, **kwargs):
    del args, kwargs
    return 200, 12000


class SourceUnitTests(unittest.TestCase):
    def test_anchors_and_units_reconstruct_exact_source(self):
        source = "Intro. Next sentence.\n\n### Step 2\nResult.\n"
        anchors = make_boundary_anchors(source)
        self.assertEqual(source, "".join(item["source_text"] for item in anchors))
        response = f"{OPEN_MARKER}\n{anchors[-1]['anchor_id']}\n{CLOSE_MARKER}"
        units = source_units_from_boundaries(
            source, anchors, parse_boundaries(response, anchors),
        )
        self.assertEqual(source, "".join(item["source_text"] for item in units))

    def test_rejects_disordered_duplicate_unknown_and_nonfinal_boundaries(self):
        anchors = make_boundary_anchors("One. Two. Three.")
        self.assertGreaterEqual(len(anchors), 3)
        invalid_bodies = [
            [anchors[1]["anchor_id"], anchors[0]["anchor_id"], anchors[-1]["anchor_id"]],
            [anchors[0]["anchor_id"], anchors[0]["anchor_id"], anchors[-1]["anchor_id"]],
            ["B9999", anchors[-1]["anchor_id"]],
            [anchors[0]["anchor_id"]],
        ]
        for body in invalid_bodies:
            with self.subTest(body=body), self.assertRaises(SourceUnitError):
                parse_boundaries(
                    OPEN_MARKER + "\n" + "\n".join(body) + "\n" + CLOSE_MARKER,
                    anchors,
                )

    def test_rejects_text_outside_exact_marker_block(self):
        anchors = make_boundary_anchors("Only one claim.")
        response = f"prose\n{OPEN_MARKER}\n{anchors[-1]['anchor_id']}\n{CLOSE_MARKER}"
        with self.assertRaises(SourceUnitError):
            parse_boundaries(response, anchors)


class SemanticIRTests(unittest.TestCase):
    def test_all_claim_forms_and_open_relation_are_valid(self):
        ir = SemanticIR.model_validate(valid_ir(), strict=True)
        self.assertEqual("is congruent to modulo seven", ir.nodes[0].claim.relation)
        self.assertEqual(["relation", "predicate", "proposition"], [
            node.claim.form for node in ir.nodes
        ])

    def test_rejects_duplicate_ids_forward_deps_definition_deps_and_bad_theorem(self):
        variants = []
        duplicate = valid_ir()
        duplicate["nodes"][1]["id"] = "n_relation"
        variants.append(duplicate)
        forward = valid_ir()
        forward["nodes"][0]["depends_on"] = ["n_predicate"]
        variants.append(forward)
        definition_dep = valid_ir()
        definition_dep["nodes"][0]["depends_on"] = ["sample_space"]
        variants.append(definition_dep)
        bad_theorem = valid_ir()
        bad_theorem["nodes"][0]["kind"] = "theorem"
        variants.append(bad_theorem)
        for payload in variants:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                SemanticIR.model_validate(payload, strict=True)


class ConversationTests(unittest.TestCase):
    def test_preserves_content_reasoning_raw_usage_and_request(self):
        response = FakeResponse("verbatim output", "verbatim thinking")
        client = FakeClient([response])
        request = {
            "model": "fake", "messages": [{"role": "user", "content": "complete input"}],
            "temperature": 0.6, "max_completion_tokens": 12,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        }
        returned, artifact = capture_chat_once(client, "test", request)
        self.assertIs(returned, response)
        self.assertEqual("complete input", artifact["request"]["messages"][0]["content"])
        self.assertEqual("verbatim thinking", artifact["assistant_reasoning_content"])
        self.assertEqual("verbatim output", artifact["assistant_content"])
        self.assertEqual("verbatim output", artifact["raw_response"]["choices"][0]["message"]["content"])
        self.assertEqual("stop", artifact["finish_reason"])
        self.assertEqual(30, artifact["usage"]["total_tokens"])

    def test_missing_reasoning_is_null_and_exception_is_captured(self):
        _, artifact = capture_chat_once(
            FakeClient([FakeResponse("output", reasoning=None)]), "test", {"messages": []},
        )
        self.assertIsNone(artifact["assistant_reasoning_content"])
        response, failed = capture_chat_once(
            FakeClient([RuntimeError("request broke")]), "test", {"messages": []},
        )
        self.assertIsNone(response)
        self.assertEqual("RuntimeError", failed["exception"]["type"])
        self.assertEqual("request broke", failed["exception"]["message"])

    def test_qwen_reasoning_alias_is_preserved(self):
        _, artifact = capture_chat_once(
            FakeClient([
                FakeResponse("output", "qwen thinking", reasoning_field="reasoning")
            ]),
            "test",
            {"messages": []},
        )
        self.assertEqual("qwen thinking", artifact["assistant_reasoning_content"])


class PipelineTests(unittest.TestCase):
    def outcomes(self):
        return [
            FakeResponse(f"{OPEN_MARKER}\nB0002\n{CLOSE_MARKER}"),
            FakeResponse(json.dumps(valid_ir())),
            FakeResponse(LEAN),
        ]

    def run_case(self, outcomes, compiler_result=None):
        client = FakeClient(outcomes)
        compiler = FakeCompiler(
            compiler_result or CompilerResult(success=True, timings={"code_sha256": "forbidden"})
        )
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        record_dir = Path(temp.name) / "record"
        result = run_record(
            RECORD, config(), record_dir, client=client, compiler=compiler,
            budgeter=budgeter,
        )
        return result, client, compiler, record_dir

    def test_success_is_exactly_three_calls_and_one_compile(self):
        result, client, compiler, record_dir = self.run_case(self.outcomes())
        self.assertEqual("completed", result["status"])
        self.assertEqual(3, len(client.calls))
        self.assertEqual(1, len(compiler.calls))
        self.assertTrue(all("tools" not in request for request in client.calls))
        self.assertEqual([12000, 12000, 12000], [
            request["max_completion_tokens"] for request in client.calls
        ])
        self.assertEqual([True, True, True], [
            request["extra_body"]["chat_template_kwargs"]["enable_thinking"]
            for request in client.calls
        ])
        self.assertNotIn("Problem", client.calls[2]["messages"][1]["content"])
        self.assertNotIn(RECORD["informal_proof"], client.calls[2]["messages"][1]["content"])
        expected = [
            "input.json", "source_split/boundary_inventory.json",
            "source_split/conversation.json", "source_units.json",
            "semantic_ir/conversation.json", "semantic_ir/raw_response.txt",
            "semantic_ir/semantic_ir.json", "blueprint/conversation.json",
            "blueprint/raw_response.txt", "blueprint/blueprint.lean",
            "blueprint/lean_result.json", "conversations.jsonl", "result.json",
        ]
        for relative in expected:
            self.assertTrue((record_dir / relative).is_file(), relative)
        self.assertEqual(3, len((record_dir / "conversations.jsonl").read_text().splitlines()))
        all_artifacts = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in record_dir.rglob("*") if path.is_file()
        )
        self.assertNotIn("sha256", all_artifacts.lower())

    def test_each_stage_failure_stops_immediately_without_retry(self):
        cases = [
            ("source_split_request_failed", [RuntimeError("split request")], 1, 0),
            ("source_split_parse_failed", [FakeResponse("bad split")], 1, 0),
            (
                "semantic_ir_request_failed",
                [self.outcomes()[0], RuntimeError("ir request")], 2, 0,
            ),
            (
                "semantic_ir_parse_failed",
                [self.outcomes()[0], FakeResponse("not json")], 2, 0,
            ),
            (
                "semantic_ir_validation_failed",
                [self.outcomes()[0], FakeResponse(json.dumps({"definitions": [], "nodes": []}))],
                2, 0,
            ),
            (
                "blueprint_request_failed",
                [self.outcomes()[0], self.outcomes()[1], RuntimeError("blueprint request")],
                3, 0,
            ),
            (
                "blueprint_extract_failed",
                [self.outcomes()[0], self.outcomes()[1], FakeResponse("not Lean")],
                3, 0,
            ),
        ]
        for expected, outcomes, calls, compile_calls in cases:
            with self.subTest(expected=expected):
                result, client, compiler, _ = self.run_case(outcomes)
                self.assertEqual(expected, result["status"])
                self.assertEqual(calls, len(client.calls))
                self.assertEqual(compile_calls, len(compiler.calls))

    def test_compile_failure_is_terminal_and_fully_saved(self):
        failure = CompilerResult(
            success=False,
            errors=["Lean error"],
            raw_output="complete Kimina output",
            failure_kind="lean",
            timings={"client_attempts": 1, "code_sha256": "must not be saved"},
        )
        result, client, compiler, record_dir = self.run_case(
            self.outcomes(), compiler_result=failure,
        )
        self.assertEqual("lean_compile_failed", result["status"])
        self.assertEqual(3, len(client.calls))
        self.assertEqual(1, len(compiler.calls))
        saved = json.loads((record_dir / "blueprint/lean_result.json").read_text())
        self.assertFalse(saved["success"])
        self.assertEqual("complete Kimina output", saved["raw_output"])
        self.assertNotIn("code_sha256", saved["timings"])


if __name__ == "__main__":
    unittest.main()
