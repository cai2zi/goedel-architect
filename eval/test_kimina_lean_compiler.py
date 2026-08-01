from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kimina_lean_compiler import CompileRequest, KiminaLeanCompiler  # noqa: E402


class FakeClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.payloads: list[dict] = []

    def post(self, url: str, json: dict) -> httpx.Response:
        self.payloads.append(json)
        return self.responses.pop(0)

    def close(self) -> None:
        pass


def response(data: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=data, request=httpx.Request("POST", "http://kimina/api/check"))


def success_item(request_id: str, *, sorries: list | None = None) -> dict:
    return {
        "id": request_id,
        "response": {"env": 1, "messages": [], "sorries": sorries or []},
    }


class KiminaLeanCompilerTest(unittest.TestCase):
    def test_batch_uses_one_request_and_restores_input_order(self) -> None:
        client = FakeClient([response({
            "results": [success_item("b"), success_item("a")],
        })])
        compiler = KiminaLeanCompiler(_client=client)
        results = compiler.check_many([
            CompileRequest("example : True := by trivial", request_id="a"),
            CompileRequest("example : 1 = 1 := by rfl", request_id="b"),
        ])

        self.assertEqual([item.success for item in results], [True, True])
        self.assertEqual(len(client.payloads), 1)
        self.assertEqual(
            [snippet["id"] for snippet in client.payloads[0]["snippets"]],
            ["a", "b"],
        )

    def test_error_keeps_position_and_end_position(self) -> None:
        diagnostic = {
            "severity": "error",
            "pos": {"line": 12, "column": 2},
            "endPos": {"line": 12, "column": 6},
            "data": "No goals to be solved",
        }
        client = FakeClient([response({
            "results": [{"id": "x", "response": {"messages": [diagnostic]}}],
        })])
        result = KiminaLeanCompiler(_client=client).check_many([
            CompileRequest("bad", request_id="x"),
        ])[0]

        self.assertFalse(result.success)
        self.assertEqual(result.failure_kind, "lean")
        parsed = json.loads(result.errors[0])
        self.assertEqual(parsed["pos"], diagnostic["pos"])
        self.assertEqual(parsed["endPos"], diagnostic["endPos"])

    def test_sorry_is_rejected_unless_explicitly_allowed(self) -> None:
        client = FakeClient([response({"results": [success_item("x", sorries=[{"goal": "True"}])]}), response({"results": [success_item("y", sorries=[{"goal": "True"}])]})])
        compiler = KiminaLeanCompiler(_client=client)
        rejected = compiler.check_many([
            CompileRequest("theorem x : True := by sorry", request_id="x"),
        ])[0]
        allowed = compiler.check_many([
            CompileRequest("theorem y : True := by sorry", True, "y"),
        ])[0]

        self.assertFalse(rejected.success)
        self.assertTrue(allowed.success)

    def test_node_assembly_failure_never_calls_server(self) -> None:
        client = FakeClient([])
        compiler = KiminaLeanCompiler(_client=client)
        result = compiler.check_node(
            "by trivial",
            node_decl="theorem root : True := by trivial",
            parent_lemma_decls="",
            header="import Mathlib",
        )
        self.assertEqual(result.failure_kind, "assembly")
        self.assertEqual(client.payloads, [])

        missing_header = compiler.check_node(
            "by trivial",
            node_decl="theorem root : True := by sorry_using []",
            parent_lemma_decls="",
            header="",
        )
        self.assertEqual(missing_header.failure_kind, "assembly")
        self.assertIn("explicit blueprint header", missing_header.errors[0])
        self.assertEqual(client.payloads, [])

    def test_safeguard_rejects_native_decide_locally(self) -> None:
        client = FakeClient([])
        result = KiminaLeanCompiler(_client=client).check(
            "example : True := by native_decide",
        )
        self.assertEqual(result.failure_kind, "assembly")
        self.assertEqual(client.payloads, [])

    def test_http_failure_is_infrastructure_error_for_every_snippet(self) -> None:
        client = FakeClient([response({"detail": "down"}, 503)])
        results = KiminaLeanCompiler(_client=client).check_many([
            CompileRequest("a", request_id="a"),
            CompileRequest("b", request_id="b"),
        ])
        self.assertEqual([item.failure_kind for item in results], ["infra", "infra"])


if __name__ == "__main__":
    unittest.main()
