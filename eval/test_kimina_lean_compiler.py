from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
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


def response(
    data: dict,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status,
        json=data,
        headers=headers,
        request=httpx.Request("POST", "http://kimina/api/check"),
    )


def success_item(request_id: str, *, sorries: list | None = None) -> dict:
    return {
        "id": request_id,
        "response": {"env": 1, "messages": [], "sorries": sorries or []},
    }


class ConcurrentClient:
    def __init__(self, delay_s: float = 0.005) -> None:
        self.delay_s = delay_s
        self.payloads: list[dict] = []
        self.active_snippets = 0
        self.peak_snippets = 0
        self.lock = threading.Lock()

    def post(self, url: str, json: dict) -> httpx.Response:
        snippets = json["snippets"]
        with self.lock:
            self.payloads.append(json)
            self.active_snippets += len(snippets)
            self.peak_snippets = max(self.peak_snippets, self.active_snippets)
        time.sleep(self.delay_s)
        with self.lock:
            self.active_snippets -= len(snippets)
        return response({
            "results": [
                {
                    "id": item["id"],
                    "response": {
                        "env": 1,
                        "messages": [{"severity": "warning", "data": item["id"]}],
                        "sorries": [],
                    },
                }
                for item in reversed(snippets)
            ],
        })

    def close(self) -> None:
        pass


class KiminaLeanCompilerTest(unittest.TestCase):
    def test_global_batcher_namespaces_duplicate_explicit_ids(self) -> None:
        client = ConcurrentClient(delay_s=0.01)
        compiler = KiminaLeanCompiler(
            _client=client,
            batch_size=8,
            global_batching=True,
            batch_wait_ms=20,
        )
        try:
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(
                    lambda index: compiler.check_many([
                        CompileRequest(
                            f"example : {index} = {index} := by rfl",
                            request_id="phase2-contract-1-0-n1",
                        ),
                    ])[0],
                    range(8),
                ))
        finally:
            compiler.close()

        self.assertTrue(all(result.success for result in results))
        self.assertEqual(len(client.payloads), 1)
        ids = [item["id"] for item in client.payloads[0]["snippets"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(
            item.startswith("phase2-contract-1-0-n1-") for item in ids
        ))

    def test_global_batcher_combines_cross_thread_singletons(self) -> None:
        client = ConcurrentClient(delay_s=0.01)
        compiler = KiminaLeanCompiler(
            _client=client,
            max_inflight_snippets=48,
            batch_size=8,
            global_batching=True,
            parallel_batches=6,
            batch_wait_ms=20,
        )
        try:
            with ThreadPoolExecutor(max_workers=96) as executor:
                results = list(executor.map(
                    lambda index: compiler.check(
                        f"example : {index} = {index} := by rfl"
                    ),
                    range(96),
                ))
            stats = compiler.stats()
        finally:
            compiler.close()

        self.assertTrue(all(result.success for result in results))
        self.assertEqual([len(payload["snippets"]) for payload in client.payloads], [8] * 12)
        self.assertLessEqual(client.peak_snippets, 48)
        self.assertEqual(stats["parallel_batches"], 6)
        self.assertEqual(stats["batch_size_distribution"], {"8": 12})
        self.assertTrue(all("micro_batch_wait_ms" in result.timings for result in results))
        self.assertTrue(all("client_http_ms" in result.timings for result in results))

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
        compiler = KiminaLeanCompiler(_client=client, retry_delays_s=())
        try:
            results = compiler.check_many([
                CompileRequest("a", request_id="a"),
                CompileRequest("b", request_id="b"),
            ])
        finally:
            compiler.close()
        self.assertEqual([item.failure_kind for item in results], ["infra", "infra"])

    def test_large_batch_is_chunked_weighted_and_ordered(self) -> None:
        client = ConcurrentClient()
        compiler = KiminaLeanCompiler(
            _client=client,
            max_inflight_snippets=128,
            batch_size=8,
        )
        requests = [
            CompileRequest(f"example : {index} = {index} := by rfl", request_id=str(index))
            for index in range(400)
        ]
        try:
            results = compiler.check_many(requests, batch_concurrency=50)
            stats = compiler.stats()
        finally:
            compiler.close()

        self.assertTrue(all(result.success for result in results))
        self.assertEqual(len(results), 400)
        self.assertEqual(
            [json.loads(result.warnings[0])["data"] for result in results],
            [str(index) for index in range(400)],
        )
        self.assertEqual(len(client.payloads), 50)
        self.assertTrue(all(len(payload["snippets"]) <= 8 for payload in client.payloads))
        self.assertLessEqual(client.peak_snippets, 128)
        self.assertEqual(stats["peak_inflight_snippets"], client.peak_snippets)
        self.assertEqual(stats["submitted_snippets"], 400)
        self.assertEqual(stats["batch_size_distribution"], {"8": 50})
        self.assertEqual(stats["current_inflight_snippets"], 0)

    def test_seventeen_requests_use_three_batches_and_keep_order(self) -> None:
        client = ConcurrentClient()
        compiler = KiminaLeanCompiler(_client=client, batch_size=8)
        try:
            results = compiler.check_many([
                CompileRequest("example : True := by trivial", request_id=str(index))
                for index in range(17)
            ], batch_concurrency=3)
        finally:
            compiler.close()

        self.assertEqual(sorted(len(payload["snippets"]) for payload in client.payloads), [1, 8, 8])
        self.assertEqual(len(results), 17)
        self.assertTrue(all(result.success for result in results))
        self.assertEqual(
            [json.loads(result.warnings[0])["data"] for result in results],
            [str(index) for index in range(17)],
        )

    def test_429_honors_retry_after_and_jitter(self) -> None:
        sleeps: list[float] = []
        client = FakeClient([
            response({"detail": "busy"}, 429, {"Retry-After": "12"}),
            response({"results": [success_item("x")]}),
        ])
        compiler = KiminaLeanCompiler(
            _client=client,
            retry_delays_s=(5,),
            retry_jitter_s=2,
            _sleep=sleeps.append,
            _random=lambda: 0.25,
        )
        try:
            result = compiler.check_many([
                CompileRequest("example : True := by trivial", request_id="x"),
            ])[0]
            stats = compiler.stats()
        finally:
            compiler.close()

        self.assertTrue(result.success)
        self.assertEqual(sleeps, [12.5])
        self.assertEqual(stats["http_requests"], 2)
        self.assertEqual(stats["http_429"], 1)
        self.assertEqual(stats["retries"], 1)

    def test_500_is_retried_as_transient_repl_failure(self) -> None:
        sleeps: list[float] = []
        client = FakeClient([
            response({"detail": "REPL exited"}, 500),
            response({"results": [success_item("x")]}),
        ])
        compiler = KiminaLeanCompiler(
            _client=client,
            retry_delays_s=(1,),
            retry_jitter_s=2,
            _sleep=sleeps.append,
            _random=lambda: 0.25,
        )
        try:
            result = compiler.check_many([
                CompileRequest("example : True := by trivial", request_id="x"),
            ])[0]
            stats = compiler.stats()
        finally:
            compiler.close()

        self.assertTrue(result.success)
        self.assertEqual(sleeps, [1.5])
        self.assertEqual(stats["http_requests"], 2)
        self.assertEqual(stats["http_5xx"], 1)
        self.assertEqual(stats["retries"], 1)

    def test_raw_output_hashes_code_instead_of_duplicating_it(self) -> None:
        code = "example : True := by trivial -- unique-secret-source"
        client = FakeClient([response({"results": [success_item("x")]})])
        compiler = KiminaLeanCompiler(_client=client)
        try:
            result = compiler.check_many([
                CompileRequest(code, request_id="x"),
            ])[0]
        finally:
            compiler.close()

        raw = json.loads(result.raw_output)
        snippet = raw["request"]["snippets"][0]
        self.assertEqual(snippet["code_chars"], len(code))
        self.assertEqual(len(snippet["code_sha256"]), 64)
        self.assertNotIn("unique-secret-source", result.raw_output)


if __name__ == "__main__":
    unittest.main()
