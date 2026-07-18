from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kimina_lean_compiler import KiminaLeanCompiler  # noqa: E402


def _command_response(
    *,
    messages: list[dict[str, Any]] | None = None,
    sorries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "results": [
            {
                "id": "test",
                "response": {
                    "env": 1,
                    "messages": messages or [],
                    "sorries": sorries or [],
                },
                "diagnostics": {"repl_uuid": "server-owned"},
            }
        ]
    }


class FakeResponse:
    def __init__(self, status_code: int, data: Any):
        self.status_code = status_code
        self._data = data
        self.text = json.dumps(data)

    def json(self) -> Any:
        return self._data


class FakeClient:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def post(self, url: str, *, json: dict[str, Any]) -> FakeResponse:
        self.calls.append((url, json))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


class KiminaLeanCompilerTest(unittest.TestCase):
    def make_compiler(
        self,
        responses: list[Any],
        *,
        sleeps: list[float] | None = None,
        concurrency: int = 8,
    ) -> tuple[KiminaLeanCompiler, FakeClient]:
        client = FakeClient(responses)
        compiler = KiminaLeanCompiler(
            _client=client,
            _sleep=(sleeps.append if sleeps is not None else lambda _delay: None),
            check_concurrency=concurrency,
        )
        return compiler, client

    def test_valid_response_and_request_shape(self) -> None:
        compiler, client = self.make_compiler([FakeResponse(200, _command_response())])
        result = compiler._run_lean("#check Nat")

        self.assertTrue(result.success)
        self.assertTrue(result.validated)
        url, payload = client.calls[0]
        self.assertEqual(url, "http://localhost:8000/api/check")
        self.assertEqual(payload["snippets"][0]["code"], "#check Nat")
        self.assertEqual(payload["timeout"], 300)
        self.assertTrue(payload["reuse"])
        self.assertFalse(payload["debug"])
        self.assertEqual(json.loads(result.raw_output)["response"], _command_response())

    def test_lean_error_is_validated_and_not_retried(self) -> None:
        data = _command_response(messages=[{"severity": "error", "data": "type mismatch"}])
        compiler, client = self.make_compiler([FakeResponse(200, data)])

        result = compiler._run_lean("bad")

        self.assertFalse(result.success)
        self.assertTrue(result.validated)
        self.assertEqual(result.errors, ["type mismatch"])
        self.assertEqual(len(client.calls), 1)

    def test_sorry_is_rejected_by_check_but_allowed_in_blueprint(self) -> None:
        sorry = [{"goal": "True", "pos": {"line": 1, "column": 1}}]
        data = _command_response(sorries=sorry)
        compiler, client = self.make_compiler(
            [FakeResponse(200, data), FakeResponse(200, data)]
        )

        proof = compiler.check("theorem t : True := by sorry")
        blueprint = compiler.check_blueprint(
            "import Mathlib\nimport Architect\n\n"
            "@[blueprint]\ntheorem t : True := by sorry_using []",
            "t",
        )

        self.assertFalse(proof.success)
        self.assertTrue(proof.has_sorry)
        self.assertTrue(blueprint.success)
        self.assertTrue(blueprint.has_sorry)
        self.assertIn("#validate_blueprint t", client.calls[1][1]["snippets"][0]["code"])

    def test_top_level_timeout_is_unvalidated(self) -> None:
        data = {"results": [{"id": "test", "error": "Lean REPL command timed out"}]}
        compiler, client = self.make_compiler([FakeResponse(200, data)])

        result = compiler._run_lean("#check Nat")

        self.assertFalse(result.success)
        self.assertFalse(result.validated)
        self.assertIn("timed out", result.errors[0])
        self.assertEqual(len(client.calls), 1)

    def test_http_timeout_is_unvalidated_and_not_retried(self) -> None:
        compiler, client = self.make_compiler([httpx.ReadTimeout("read timed out")])

        result = compiler._run_lean("#check Nat")

        self.assertFalse(result.success)
        self.assertFalse(result.validated)
        self.assertIn("read timed out", result.errors[0])
        self.assertEqual(len(client.calls), 1)

    def test_429_retries_three_times_with_required_backoff(self) -> None:
        sleeps: list[float] = []
        responses = [
            FakeResponse(429, {"detail": "No available REPLs"}),
            FakeResponse(429, {"detail": "No available REPLs"}),
            FakeResponse(429, {"detail": "No available REPLs"}),
            FakeResponse(200, _command_response()),
        ]
        compiler, client = self.make_compiler(responses, sleeps=sleeps)

        result = compiler._run_lean("#check Nat")

        self.assertTrue(result.success)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(sleeps, [0.5, 1.0, 2.0])

    def test_no_available_repl_result_retries(self) -> None:
        first = {"results": [{"id": "test", "error": "No available REPLs"}]}
        compiler, client = self.make_compiler(
            [FakeResponse(200, first), FakeResponse(200, _command_response())]
        )

        result = compiler._run_lean("#check Nat")

        self.assertTrue(result.success)
        self.assertEqual(len(client.calls), 2)

    def test_http_500_and_disconnect_are_unvalidated(self) -> None:
        compiler_500, client_500 = self.make_compiler(
            [FakeResponse(500, {"detail": "server exploded"})]
        )
        disconnected = httpx.ConnectError("disconnected")
        compiler_net, client_net = self.make_compiler([disconnected])

        result_500 = compiler_500._run_lean("#check Nat")
        result_net = compiler_net._run_lean("#check Nat")

        self.assertFalse(result_500.validated)
        self.assertIn("HTTP 500", result_500.errors[0])
        self.assertEqual(len(client_500.calls), 1)
        self.assertFalse(result_net.validated)
        self.assertIn("disconnected", result_net.errors[0])
        self.assertEqual(len(client_net.calls), 1)

    def test_bounded_semaphore_caps_concurrent_posts(self) -> None:
        class ConcurrentClient:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.active = 0
                self.max_active = 0

            def post(self, _url: str, *, json: dict[str, Any]) -> FakeResponse:
                del json
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.02)
                with self.lock:
                    self.active -= 1
                return FakeResponse(200, _command_response())

            def close(self) -> None:
                pass

        client = ConcurrentClient()
        compiler = KiminaLeanCompiler(_client=client, check_concurrency=2)
        threads = [threading.Thread(target=compiler._run_lean, args=(f"#check {i}",)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(client.max_active, 2)

    def test_close_only_closes_http_client(self) -> None:
        compiler, client = self.make_compiler([])
        compiler.close()
        compiler.close()
        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()
