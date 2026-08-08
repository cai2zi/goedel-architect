from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from cot_blueprint_refine.common import read_jsonl  # noqa: E402
from cot_blueprint_refine.replay_refinement_outputs import (  # noqa: E402
    replay_refinement_outputs,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _response(content: str, finish_reason: str, *, prompt: int, completion: int) -> dict:
    return {
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"content": content},
        }],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def _event(
    attempt: int,
    content: str,
    *,
    finish_reason: str = "stop",
    mode: str = "primary",
    prompt: int = 10,
    completion: int = 5,
) -> dict:
    return {
        "attempt": attempt,
        "request_mode": mode,
        "status": "ok",
        "latency_s": float(attempt),
        "request": {
            "base_url": "http://unused.invalid/v1",
            "model": "persisted-model",
            "messages": [{"role": "user", "content": "persisted prompt"}],
            "temperature": 0.6,
            "max_tokens": 20480 if mode == "primary" else 8192,
            "timeout_s": 30,
        },
        "response": _response(
            content,
            finish_reason,
            prompt=prompt,
            completion=completion,
        ),
        "assistant_content": content,
        "assistant_reasoning_content": "persisted reasoning",
        "finish_reason": finish_reason,
    }


class RefinementReplayTest(unittest.TestCase):
    def _root(self, tmp: str, ids: list[str]) -> Path:
        root = Path(tmp) / "experiment"
        contexts = [{"ID": row_id, "problem": f"problem {row_id}"} for row_id in ids]
        current = [{
            "ID": row_id,
            "status": "invalid_output",
            "error": "old strict result",
            "refined_cot": "",
            "attempts": 2,
        } for row_id in ids]
        _write_jsonl(root / "blueprint_contexts/blueprint_contexts.jsonl", contexts)
        _write_jsonl(root / "refinement/blueprint/refined_predictions.jsonl", current)
        return root

    def _conversation(self, root: Path, row_id: str, events: list[dict]) -> None:
        path = root / "refinement/blueprint/conversations" / f"{row_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"ID": row_id, "events": events}),
            encoding="utf-8",
        )

    def test_selects_earliest_outside_envelope_and_counts_avoided_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, ["A"])
            first = _event(
                1,
                "analysis outside\n<final_refined_solution>first \\boxed{4}"
                "</final_refined_solution>",
                prompt=100,
                completion=200,
            )
            recovery = _event(
                2,
                "<final_refined_solution>recovery \\boxed{5}</final_refined_solution>",
                mode="concise_recovery",
                prompt=30,
                completion=40,
            )
            self._conversation(root, "A", [first, recovery])

            metrics = replay_refinement_outputs(root, "blueprint", "lenient-v1")
            rows = read_jsonl(Path(metrics["output"]))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["refined_cot"], "first \\boxed{4}")
            self.assertEqual(row["replay_selected_attempt"], 1)
            self.assertEqual(row["replay_selected_mode"], "primary")
            self.assertEqual(
                row["final_envelope_warning"],
                "content_outside_final_refined_solution",
            )
            self.assertEqual(row["replay_avoided_request_count"], 1)
            self.assertEqual(row["replay_avoided_total_tokens_known"], 70)
            self.assertEqual(metrics["avoided_prompt_tokens_known"], 30)
            self.assertEqual(metrics["avoided_completion_tokens_known"], 40)

    def test_multiple_markers_uses_last_complete_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, ["A"])
            content = (
                "<final_refined_solution>draft \\boxed{1}</final_refined_solution>\n"
                "<final_refined_solution>final \\boxed{2}</final_refined_solution>"
            )
            self._conversation(root, "A", [_event(1, content)])

            metrics = replay_refinement_outputs(root, "blueprint", "multiple")
            row = read_jsonl(Path(metrics["output"]))[0]

            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["refined_cot"], "final \\boxed{2}")
            self.assertEqual(
                row["final_envelope_warning"],
                "multiple_final_refined_solution_markers",
            )

    def test_length_unclosed_conflict_and_missing_envelope_are_never_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ids = ["length", "unclosed", "conflict", "missing"]
            root = self._root(tmp, ids)
            self._conversation(root, "length", [
                _event(
                    1,
                    "<final_refined_solution>cut \\boxed{1}</final_refined_solution>",
                    finish_reason="length",
                ),
                _event(
                    2,
                    "<final_refined_solution>good \\boxed{2}</final_refined_solution>",
                    mode="concise_recovery",
                ),
            ])
            self._conversation(root, "unclosed", [
                _event(1, "<final_refined_solution>cut \\boxed{1}"),
            ])
            self._conversation(root, "conflict", [
                _event(
                    1,
                    "<final_refined_solution>\\boxed{1} then \\boxed{2}"
                    "</final_refined_solution>",
                ),
            ])
            self._conversation(root, "missing", [_event(1, "bare answer \\boxed{1}")])

            metrics = replay_refinement_outputs(root, "blueprint", "hard-rejections")
            rows = {row["ID"]: row for row in read_jsonl(Path(metrics["output"]))}

            self.assertEqual(rows["length"]["replay_selected_attempt"], 2)
            self.assertEqual(rows["length"]["refined_cot"], "good \\boxed{2}")
            for row_id in ("unclosed", "conflict", "missing"):
                self.assertEqual(
                    rows[row_id]["replay_strategy"],
                    "retained_latest_terminal",
                )
                self.assertEqual(rows[row_id]["status"], "invalid_output")
            errors = {
                row_id: rows[row_id]["replay_attempt_diagnostics"][0][
                    "normalization_error"
                ]
                for row_id in ("unclosed", "conflict", "missing")
            }
            self.assertEqual(errors["unclosed"], "unclosed_final_refined_solution")
            self.assertTrue(errors["conflict"].startswith("conflicting_boxed_answers"))
            self.assertEqual(errors["missing"], "missing_final_refined_solution_open")

    def test_retains_current_row_without_conversation_and_never_reuses_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, ["A"])
            original_path = root / "refinement/blueprint/refined_predictions.jsonl"
            original = original_path.read_bytes()

            metrics = replay_refinement_outputs(root, "blueprint", "immutable")
            row = read_jsonl(Path(metrics["output"]))[0]

            self.assertEqual(row["replay_strategy"], "retained_latest_terminal")
            self.assertEqual(original_path.read_bytes(), original)
            self.assertNotEqual(Path(metrics["output"]), original_path)
            with self.assertRaises(FileExistsError):
                replay_refinement_outputs(root, "blueprint", "immutable")

    def test_records_unknown_tokens_for_a_later_failed_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, ["A"])
            first = _event(
                1,
                "outside<final_refined_solution>ok \\boxed{1}</final_refined_solution>",
            )
            later = {
                "attempt": 2,
                "request_mode": "concise_recovery",
                "status": "exception",
                "request": {"model": "persisted-model", "messages": []},
                "response": None,
                "exception": {"type": "APIConnectionError"},
            }
            self._conversation(root, "A", [first, later])

            metrics = replay_refinement_outputs(root, "blueprint", "unknown-token")
            row = read_jsonl(Path(metrics["output"]))[0]

            self.assertEqual(row["replay_avoided_request_count"], 1)
            self.assertEqual(
                row["replay_avoided_requests_with_unknown_total_tokens"],
                1,
            )
            self.assertEqual(metrics["avoided_requests_with_unknown_total_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
