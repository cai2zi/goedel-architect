"""Tests for dynamically exposing repo_search in the Phase 2 prover."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lean_compiler import CompilerResult  # noqa: E402
from prover import GoedelProver, ProofSignal  # noqa: E402


def _tool_response(name: str, arguments: dict, call_id: str) -> SimpleNamespace:
    tool_call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=None,
    )


class _RecordingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.responses = [
            _tool_response(
                "mathlib_search",
                {"query": "True introduction", "k": 5},
                "search-call",
            ),
            _tool_response(
                "lean_compile",
                {"proof_body": ":= by trivial"},
                "compile-call",
            ),
        ]

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self) -> None:
        self.completions = _RecordingCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class _FakeCompiler:
    def check(self, _proof_body: str, **_kwargs) -> CompilerResult:
        return CompilerResult(success=True)


class _FakeMathlibRetrieval:
    def search(self, _query: str, _k: int = 10) -> list:
        return []


class ProverDynamicToolsTest(unittest.TestCase):
    def _run(self, repo_retrieval):
        client = _FakeClient()
        with patch("prover.make_client", return_value=client):
            prover = GoedelProver(
                model_id="test-model",
                retrieval=_FakeMathlibRetrieval(),
            )
            result = prover.prove_node(
                compiler=_FakeCompiler(),
                node_name="test_node",
                node_stmt="theorem test_node : True := by sorry_using []",
                repo_retrieval=repo_retrieval,
            )
        return result, client.completions.calls

    def test_repo_search_is_hidden_without_repo_retrieval(self) -> None:
        result, calls = self._run(repo_retrieval=None)

        self.assertEqual(result.signal, ProofSignal.SOLVED)
        self.assertEqual(len(calls), 2)
        for call in calls:
            tool_names = {tool["function"]["name"] for tool in call["tools"]}
            self.assertEqual(tool_names, {"lean_compile", "mathlib_search"})
        system_prompt = calls[0]["messages"][0]["content"]
        self.assertIn("No repository search", system_prompt)
        self.assertIn("tool is available in this experiment", system_prompt)
        self.assertNotIn("You have three tools", system_prompt)

    def test_repo_search_is_exposed_with_repo_retrieval(self) -> None:
        result, calls = self._run(repo_retrieval=object())

        self.assertEqual(result.signal, ProofSignal.SOLVED)
        self.assertEqual(len(calls), 2)
        for call in calls:
            tool_names = {tool["function"]["name"] for tool in call["tools"]}
            self.assertEqual(
                tool_names,
                {"lean_compile", "repo_search", "mathlib_search"},
            )
        system_prompt = calls[0]["messages"][0]["content"]
        self.assertIn("You have three tools", system_prompt)
        self.assertIn("call repo_search", system_prompt)
        self.assertNotIn(
            "You have two tools, `lean_compile` and `mathlib_search`",
            system_prompt,
        )


if __name__ == "__main__":
    unittest.main()
