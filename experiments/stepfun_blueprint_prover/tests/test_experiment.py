from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]
REPO_ROOT = HERE.parents[1]
for path in (str(REPO_ROOT / "src"), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from blueprint import Blueprint, BlueprintNode
from input_loader import load_accepted_blueprints
from node_context import build_node_problem, transitive_dependencies
from run_experiment import RecordRuntime, run_negative, run_positive
from stepfun_repl_prover import (
    ProverOutcome,
    StepFunReplProver,
    check_node_safely,
    extract_proof_body,
    extract_sketch,
)
from goedel_self_correct_prover import GoedelSelfCorrectProver, GOEDEL_USER_PROMPT
from kimina_lean_compiler import CompilerResult


SOURCE_ROOT = Path(
    "/ssd/czx/czx_work/cot_blueprint_refine/"
    "qwen3_8b_397b_wrong76_whole_cot_blueprint_generation_thinking_judge/"
    "robustpa/blueprint"
)


def proof_node(name: str, deps: list[str], kind: str = "lemma") -> BlueprintNode:
    return BlueprintNode(
        name=name,
        kind=kind,
        statement=name,
        proof_sketch="",
        dependencies=deps,
        lean_declaration=(
            f"@[blueprint (statement := /-- {name} -/) (proof := /-- proof -/)]\n"
            f"{kind} {name} : True := by sorry_using [{', '.join(deps)}]"
        ),
    )


def synthetic_blueprint() -> Blueprint:
    definition = BlueprintNode(
        name="d", kind="definition", statement="", proof_sketch="",
        lean_declaration=(
            "@[blueprint (statement := /-- d -/)]\ndef d : Nat := 1"
        ),
    )
    return Blueprint(
        nodes=[definition, proof_node("l1", []), proof_node("root", ["l1"], "theorem")],
        lean_file="",
        target_theorem="root",
        phase2_header="import Mathlib\n",
    )


class ExtractionTests(unittest.TestCase):
    def test_sketch_and_full_theorem(self) -> None:
        text = "reasoning<sketch>\n```lean\nby simp\n```\n</sketch>"
        self.assertEqual(extract_proof_body(extract_sketch(text) or ""), "by simp")
        final = "</think>\n```lean\ntheorem x : True := by trivial\n```<｜end▁of▁sentence｜>"
        self.assertEqual(extract_proof_body(final), "by trivial")

    def test_complete_file_uses_last_declaration_proof(self) -> None:
        text = """```lean4
lemma parent : True := by trivial
theorem root : True := by exact True.intro
```"""
        self.assertEqual(extract_proof_body(text), "by exact True.intro")


class InputTests(unittest.TestCase):
    def test_real_source_selects_exact_strict_accepted_set(self) -> None:
        rows = load_accepted_blueprints(SOURCE_ROOT)
        self.assertEqual(len(rows), 45)
        self.assertEqual(len({row.record_id for row in rows}), 45)
        self.assertTrue(all(row.state.semantic_status == "strictAccepted" for row in rows))


class ContextTests(unittest.TestCase):
    def test_only_ancestor_proofs_are_included(self) -> None:
        blueprint = synthetic_blueprint()
        root = blueprint.node_by_name("root")
        self.assertEqual(transitive_dependencies(blueprint, root), {"l1"})
        problem = build_node_problem(blueprint, "root", {"l1": "by trivial"}, stage="positive")
        self.assertIn("def d", problem.parent_lemma_decls)
        self.assertIn("theorem l1", problem.parent_lemma_decls)
        self.assertIn("theorem root", problem.complete_lean)
        self.assertNotIn("sorry_using", problem.complete_lean)

    def test_negative_rewrites_only_conclusion(self) -> None:
        problem = build_node_problem(
            synthetic_blueprint(), "root", {"l1": "by trivial"}, stage="negative",
        )
        self.assertIn("theorem neg_root", problem.node_decl)
        self.assertIn(": ¬ (True)", problem.node_decl)

    def test_negative_closes_node_binders_before_negating(self) -> None:
        blueprint = synthetic_blueprint()
        node = blueprint.node_by_name("root")
        node.lean_declaration = (
            "@[blueprint (statement := /-- root -/) (proof := /-- proof -/)]\n"
            "theorem root (n : Nat) (h : 5 < n) : n = 7 := by sorry_using [l1]"
        )
        problem = build_node_problem(
            blueprint, "root", {"l1": "by trivial"}, stage="negative",
        )
        self.assertIn(
            "theorem neg_root : ¬ (∀ (n : Nat) (h : 5 < n), n = 7)",
            problem.node_decl,
        )
        self.assertNotIn("theorem neg_root (n", problem.node_decl)


class FakeProver:
    def __init__(self, statuses: dict[tuple[str, str], str] | None = None):
        self.statuses = statuses or {}
        self.calls: list[tuple[str, str]] = []

    async def prove(self, problem):
        self.calls.append((problem.stage, problem.node_name))
        status = self.statuses.get((problem.stage, problem.node_name), "solved")
        return ProverOutcome(status, proof_body="by trivial" if status == "solved" else "")


def synthetic_runtime(directory: Path) -> RecordRuntime:
    blueprint = synthetic_blueprint()
    source = SimpleNamespace(
        record_id="record", source_id="source", subset="subset", split="split",
        blueprint=blueprint,
    )
    data = {
        "active_nodes": ["d", "l1", "root"], "positive": {}, "negative": {},
        "proved_cache": {}, "final": {},
    }
    checkpoint = directory / "checkpoint.json"
    checkpoint.write_text(json.dumps(data))
    return RecordRuntime(source, checkpoint, data)


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_positive_releases_dependency_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = synthetic_runtime(Path(tmp))
            prover = FakeProver()
            await run_positive([runtime], prover, Path(tmp))
            self.assertEqual(prover.calls, [("positive", "l1"), ("positive", "root")])
            self.assertEqual(runtime.positive["root"]["status"], "solved")

    async def test_failed_parent_blocks_child_then_negative_probes_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = synthetic_runtime(Path(tmp))
            positive = FakeProver({("positive", "l1"): "lean_error"})
            await run_positive([runtime], positive, Path(tmp))
            self.assertEqual(runtime.positive["root"]["status"], "blocked_by_dependency")
            negative = FakeProver()
            await run_negative([runtime], negative, Path(tmp))
            self.assertEqual(negative.calls, [("negative", "l1")])
            self.assertEqual(runtime.negative["l1"]["status"], "formally_negated")


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT"

    def encode(self, prompt, add_special_tokens=False):
        return list(range(len(prompt)))


class FakeGoedelTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return list(range(sum(len(message["content"]) for message in messages)))


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            choice = SimpleNamespace(
                text="<sketch>\nby trivial\n</sketch>", finish_reason="stop",
                stop_reason=151666, model_extra={},
            )
        else:
            choice = SimpleNamespace(
                text="```lean\ntheorem root : True := by trivial\n```<｜end▁of▁sentence｜>",
                finish_reason="stop", stop_reason=151643, model_extra={},
            )
        return SimpleNamespace(
            choices=[choice],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class FakeCompiler:
    def check(self, lean_code, allow_sorry=False):
        self.last_code = lean_code
        return CompilerResult(True, raw_output="{}")


class FakeGoedelCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        number = len(self.calls)
        text = (
            "Proof plan: try the wrong tactic.\n```lean4\ntheorem root : True := by omega\n```"
            if number == 1 else
            "Updated plan: use the constructor.\n```lean4\ntheorem root : True := by trivial\n```"
        )
        message = SimpleNamespace(content=text, reasoning_content=None, model_extra={})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10),
        )


class CorrectingCompiler:
    def __init__(self):
        self.calls = 0

    def check(self, lean_code, allow_sorry=False):
        self.calls += 1
        if self.calls == 1:
            return CompilerResult(False, errors=["unknown tactic"], failure_kind="lean")
        return CompilerResult(True, raw_output="{}")

    def check_node(self, proof_body, **kwargs):
        return CompilerResult(True, raw_output="{}")


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_lean_backslash_is_literal_during_assembly(self) -> None:
        compiler = FakeCompiler()
        blueprint = synthetic_blueprint()
        result = check_node_safely(
            compiler,
            r"by simpa using Set.mem_setOf_eq",
            node_decl=blueprint.node_by_name("l1").lean_declaration,
            parent_lemma_decls="",
            header=blueprint.phase2_header,
        )
        self.assertTrue(result.success)
        self.assertIn(r"by simpa using Set.mem_setOf_eq", compiler.last_code)

    async def test_sketch_feedback_then_final(self) -> None:
        client = SimpleNamespace(completions=FakeCompletions())
        prover = StepFunReplProver(
            client=client,
            tokenizer=FakeTokenizer(),
            compiler=FakeCompiler(),
            config={
                "name": "model", "max_context_tokens": 40960,
                "temperature": 1.0, "top_p": 0.999, "top_k": -1,
                "seed": 42, "stop_token_ids": [151643, 151666],
                "include_stop_str_in_output": True, "api_concurrency": 8,
            },
        )
        problem = build_node_problem(
            synthetic_blueprint(), "root", {"l1": "by trivial"}, stage="positive",
        )
        outcome = await prover.prove(problem)
        self.assertEqual(outcome.status, "solved")
        self.assertEqual(outcome.turns, 2)
        self.assertEqual(client.completions.calls, 2)

    async def test_goedel_native_prompt_and_two_round_self_correction(self) -> None:
        completions = FakeGoedelCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        compiler = CorrectingCompiler()
        prover = GoedelSelfCorrectProver(
            client=client,
            tokenizer=FakeGoedelTokenizer(),
            compiler=compiler,
            config={
                "name": "Goedel-Prover-V2-8B", "max_context_tokens": 40960,
                "initial_max_tokens": 32768, "correction_max_tokens": 8192,
                "self_correction_rounds": 2, "temperature": 0.6,
                "top_p": 0.95, "top_k": 20, "seed": 30,
                "api_concurrency": 8,
            },
        )
        problem = build_node_problem(
            synthetic_blueprint(), "root", {"l1": "by trivial"}, stage="positive",
        )
        outcome = await prover.prove(problem)
        self.assertEqual(outcome.status, "solved")
        self.assertEqual(outcome.proof_body, "by trivial")
        self.assertEqual(outcome.turns, 2)
        self.assertIn("Complete the following Lean 4 code", completions.calls[0]["messages"][0]["content"])
        self.assertIn("unknown tactic", completions.calls[1]["messages"][-1]["content"])
        self.assertEqual(completions.calls[0]["max_tokens"], 32768)
        self.assertLessEqual(completions.calls[1]["max_tokens"], 8192)

    def test_goedel_prompt_matches_model_card_protocol(self) -> None:
        self.assertIn("Complete the following Lean 4 code", GOEDEL_USER_PROMPT)
        self.assertIn("detailed proof plan", GOEDEL_USER_PROMPT)


if __name__ == "__main__":
    unittest.main()
