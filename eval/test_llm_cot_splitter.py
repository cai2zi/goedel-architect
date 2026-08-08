from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from cot_blueprint_refine.llm_cot_splitter import (  # noqa: E402
    LLMCotSplitterConfig,
    RESULTS_FILENAME,
    SUMMARY_FILENAME,
    SplitFormatError,
    atomize_cot,
    build_split_messages,
    parse_split_boundaries,
    sections_from_boundaries,
    split_cot_rows,
)


def response(
    content: str,
    *,
    finish_reason: str = "stop",
    prompt_tokens: int = 20,
    completion_tokens: int = 5,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="chatcmpl-split-test",
        request_id="request-split-test",
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=content,
                reasoning_content="private splitter reasoning",
                model_extra={},
            ),
            finish_reason=finish_reason,
        )],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class FakeCompletions:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected LLM request")
        return self.responses.pop(0)


def client_for(responses: list[SimpleNamespace]):
    completions = FakeCompletions(responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


class AtomizationTest(unittest.TestCase):
    def test_atoms_losslessly_cover_math_code_tables_and_prose(self) -> None:
        source = """  First claim. Second claim with $x. y$ unchanged.

$$
x = y. z
$$

```lean
example : True := by
  trivial
```

| n | value |
|---|---|
| 1 | 2 |

---

Therefore the answer is 2.  """
        atoms = atomize_cot(source)

        self.assertEqual("".join(atom["source_text"] for atom in atoms), source.strip())
        self.assertEqual(atoms[0]["source_start"], len(source) - len(source.lstrip()))
        self.assertEqual(atoms[-1]["source_end"], len(source.rstrip()))
        self.assertEqual(
            [atom["source_start"] for atom in atoms[1:]],
            [atom["source_end"] for atom in atoms[:-1]],
        )
        self.assertEqual(
            [atom["atom_id"] for atom in atoms],
            [f"A{index:04d}" for index in range(1, len(atoms) + 1)],
        )
        display = [atom for atom in atoms if atom["kind"] == "display_math"]
        code = [atom for atom in atoms if atom["kind"] == "code_block"]
        tables = [atom for atom in atoms if atom["kind"] == "table_row"]
        self.assertEqual(len(display), 1)
        self.assertIn("x = y. z", display[0]["source_text"])
        self.assertEqual(len(code), 1)
        self.assertIn("example : True", code[0]["source_text"])
        self.assertEqual(len(tables), 3)
        self.assertFalse(any(atom["source_text"].strip() == "---" for atom in atoms))
        self.assertTrue(any("---" in atom["source_text"] for atom in atoms))
        inline_math_atoms = [atom for atom in atoms if "$x. y$" in atom["source_text"]]
        self.assertEqual(len(inline_math_atoms), 1)
        self.assertGreaterEqual(len([atom for atom in atoms if atom["kind"] == "prose"]), 3)

    def test_empty_source_has_no_atoms_and_non_string_is_rejected(self) -> None:
        self.assertEqual(atomize_cot(" \n\t "), [])
        with self.assertRaises(TypeError):
            atomize_cot(None)  # type: ignore[arg-type]


class BoundaryParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = "  First claim. Second claim. Third claim.  "
        self.atoms = atomize_cot(self.source)
        self.assertEqual(len(self.atoms), 3)

    def test_strict_parser_and_sections_are_exact(self) -> None:
        boundaries = parse_split_boundaries(
            "\n[[COT_SPLIT_V1]]\nA0001\nA0003\n[[/COT_SPLIT_V1]]\n",
            self.atoms,
        )
        sections = sections_from_boundaries(self.source, self.atoms, boundaries)

        self.assertEqual(boundaries, ["A0001", "A0003"])
        self.assertEqual([section["atom_ids"] for section in sections], [
            ["A0001"], ["A0002", "A0003"],
        ])
        self.assertEqual("".join(section["source_text"] for section in sections), self.source.strip())
        for section in sections:
            self.assertEqual(
                section["source_text"],
                self.source[section["source_start"]:section["source_end"]],
            )

    def test_extracts_unique_block_with_outside_text(self) -> None:
        boundaries = parse_split_boundaries(
            "I grouped the steps below.\n"
            "[[COT_SPLIT_V1]]\nA0001\nA0003\n[[/COT_SPLIT_V1]]\n"
            "Done.",
            self.atoms,
        )
        self.assertEqual(boundaries, ["A0001", "A0003"])

    def test_rejects_bad_ids_order_duplicates_and_missing_final(self) -> None:
        invalid = {
            "unknown": "[[COT_SPLIT_V1]]\nA9999\n[[/COT_SPLIT_V1]]",
            "decreasing": "[[COT_SPLIT_V1]]\nA0002\nA0001\nA0003\n[[/COT_SPLIT_V1]]",
            "duplicate": "[[COT_SPLIT_V1]]\nA0001\nA0001\nA0003\n[[/COT_SPLIT_V1]]",
            "missing final": "[[COT_SPLIT_V1]]\nA0002\n[[/COT_SPLIT_V1]]",
            "comma list": "[[COT_SPLIT_V1]]\nA0001,A0003\n[[/COT_SPLIT_V1]]",
            "two blocks": (
                "[[COT_SPLIT_V1]]\nA0003\n[[/COT_SPLIT_V1]]\n"
                "[[COT_SPLIT_V1]]\nA0003\n[[/COT_SPLIT_V1]]"
            ),
        }
        for reason, content in invalid.items():
            with self.subTest(reason=reason):
                with self.assertRaises(SplitFormatError):
                    parse_split_boundaries(content, self.atoms)

    def test_tampered_atom_inventory_is_rejected_before_reconstruction(self) -> None:
        tampered = [dict(atom) for atom in self.atoms]
        tampered[1]["source_text"] = "silently rewritten"
        with self.assertRaisesRegex(ValueError, "source text mismatch"):
            sections_from_boundaries(self.source, tampered, ["A0003"])

    def test_canonicalizes_heading_only_and_intro_formula_boundaries(self) -> None:
        source = "### Compute\nThe value is:\n$$\nx = 2\n$$\nTherefore done."
        atoms = atomize_cot(source)
        heading = next(atom for atom in atoms if atom["kind"] == "heading")
        intro = next(atom for atom in atoms if atom["source_text"].rstrip().endswith(":"))
        final = atoms[-1]["atom_id"]

        self.assertEqual(
            parse_split_boundaries(
                f"[[COT_SPLIT_V1]]\n{heading['atom_id']}\n{final}\n[[/COT_SPLIT_V1]]",
                atoms,
            ),
            [final],
        )
        self.assertEqual(
            parse_split_boundaries(
                f"[[COT_SPLIT_V1]]\n{intro['atom_id']}\n{final}\n[[/COT_SPLIT_V1]]",
                atoms,
            ),
            [final],
        )

    def test_moves_embedded_heading_boundary_before_heading(self) -> None:
        source = (
            "Previous computation gives x = 1.\n"
            "### Step 2: New computation\n"
            "Now y = 2."
        )
        atoms = atomize_cot(source)
        heading = next(atom for atom in atoms if atom["kind"] == "heading")
        final = atoms[-1]["atom_id"]

        boundaries = parse_split_boundaries(
            f"[[COT_SPLIT_V1]]\n{heading['atom_id']}\n{final}\n[[/COT_SPLIT_V1]]",
            atoms,
        )

        self.assertEqual(boundaries, [atoms[0]["atom_id"], final])
        sections = sections_from_boundaries(source, atoms, boundaries)
        self.assertEqual(sections[0]["source_text"].strip(), "Previous computation gives x = 1.")
        self.assertTrue(sections[1]["source_text"].lstrip().startswith("### Step 2"))

    def test_moves_embedded_colon_intro_boundary_before_intro(self) -> None:
        source = "Previous result is x = 1.\nNow compute:\n$$\ny = 2\n$$"
        atoms = atomize_cot(source)
        intro = next(atom for atom in atoms if atom["source_text"].rstrip().endswith(":"))
        final = atoms[-1]["atom_id"]

        boundaries = parse_split_boundaries(
            f"[[COT_SPLIT_V1]]\n{intro['atom_id']}\n{final}\n[[/COT_SPLIT_V1]]",
            atoms,
        )

        self.assertEqual(boundaries, [atoms[0]["atom_id"], final])

    def test_does_not_shift_intro_boundary_onto_numbered_title(self) -> None:
        source = (
            "### Observations\n"
            "1. **Internal Angles**:\n"
            "The angle is:\n"
            "$$\nx = 2\n$$"
        )
        atoms = atomize_cot(source)
        title = next(atom for atom in atoms if "Internal Angles" in atom["source_text"])
        intro = next(atom for atom in atoms if "The angle is:" in atom["source_text"])
        final = atoms[-1]["atom_id"]

        boundaries = parse_split_boundaries(
            "[[COT_SPLIT_V1]]\n"
            f"{title['atom_id']}\n{intro['atom_id']}\n{final}\n"
            "[[/COT_SPLIT_V1]]",
            atoms,
        )

        self.assertEqual(boundaries, [final])

    def test_canonicalizes_blank_key_table_continuation_rows(self) -> None:
        source = "| n | k |\n|---|---|\n| 2 | 1 |\n|   | 2 |\n| 3 | 1 |"
        atoms = atomize_cot(source)
        keyed_two = atoms[2]["atom_id"]
        continuation = atoms[3]["atom_id"]
        keyed_three = atoms[4]["atom_id"]

        self.assertEqual(
            parse_split_boundaries(
                "[[COT_SPLIT_V1]]\n"
                f"{keyed_two}\n{continuation}\n{keyed_three}\n"
                "[[/COT_SPLIT_V1]]",
                atoms,
            ),
            [continuation, keyed_three],
        )


class PromptTest(unittest.TestCase):
    def test_prompt_limits_model_to_boundaries_and_preserves_wrong_cot(self) -> None:
        source = "An incorrect step says x = 3. Therefore x = 4."
        messages = build_split_messages(source, atomize_cot(source))
        prompt = "\n".join(message["content"] for message in messages)
        normalized = " ".join(prompt.split())

        self.assertIn("may be mathematically wrong", prompt)
        self.assertIn("do not solve, repair, summarize, translate, or rewrite", prompt)
        self.assertIn("smallest self-contained unit", prompt)
        self.assertIn("at most one independently checkable", prompt)
        self.assertIn("each substantive list item must end its own step", prompt)
        self.assertIn("not as one step per physical row", prompt)
        self.assertIn("not authoritative boundaries", prompt)
        self.assertIn("last atom must be a boundary", prompt)
        self.assertIn('"atom_id": "A0001"', prompt)
        self.assertIn("exactly one block", normalized)
        self.assertNotIn("FINAL_ATOM_ID", prompt)
        self.assertIn("final boundary must literally be A0002", normalized)


class SplitRowsTest(unittest.TestCase):
    def _config(self, **overrides) -> LLMCotSplitterConfig:
        values = {
            "model": "test-model",
            "openai_base_url": "http://localhost:8001/v1",
            "concurrency": 2,
            "max_tokens": 128,
            "max_format_attempts": 2,
            "enable_thinking": False,
        }
        values.update(overrides)
        return LLMCotSplitterConfig(**values)

    def test_concurrent_rows_persist_full_artifacts_metrics_and_request_contract(self) -> None:
        rows = [
            {"name": "row-a", "post_think_cot": "First. Second."},
            {"name": "row-b", "post_think_cot": "Alpha. Beta."},
        ]
        valid = (
            "Here are the boundaries.\n"
            "[[COT_SPLIT_V1]]\nA0001\nA0002\n[[/COT_SPLIT_V1]]\nDone."
        )
        client, completions = client_for([response(valid), response(valid)])
        with tempfile.TemporaryDirectory() as directory:
            results = asyncio.run(split_cot_rows(
                rows, self._config(), directory, client=client,
            ))
            artifact_lines = [
                json.loads(line)
                for line in (Path(directory) / RESULTS_FILENAME).read_text(encoding="utf-8").splitlines()
            ]
            summary = json.loads(
                (Path(directory) / SUMMARY_FILENAME).read_text(encoding="utf-8")
            )

        self.assertEqual(list(results), ["row-a", "row-b"])
        self.assertTrue(all(result.ok for result in results.values()))
        self.assertEqual(len(completions.calls), 2)
        for request in completions.calls:
            self.assertNotIn("response_format", request)
            self.assertEqual(request["temperature"], 0.0)
            self.assertEqual(
                request["extra_body"],
                {"chat_template_kwargs": {"enable_thinking": False}},
            )
        self.assertEqual(len(artifact_lines), 2)
        for artifact in artifact_lines:
            self.assertTrue(artifact["ID"])
            self.assertTrue(artifact["source_sha256"])
            self.assertTrue(artifact["cache_key"])
            self.assertEqual(artifact["status"], "ok")
            self.assertEqual(len(artifact["atoms"]), 2)
            self.assertEqual(len(artifact["attempts"]), 1)
            self.assertEqual(len(artifact["raw_responses"]), 1)
            self.assertEqual(
                artifact["attempts"][0]["format_warnings"],
                ["content_outside_marker_block"],
            )
            self.assertIn("Here are", artifact["attempts"][0]["outside_content"]["prefix"])
            self.assertEqual(artifact["usage"]["total_tokens"], 25)
            self.assertGreaterEqual(artifact["latency_s"], 0)
        self.assertEqual(summary["status_counts"], {"ok": 2})
        self.assertEqual(summary["request_attempts"], 2)
        self.assertEqual(summary["usage"]["total_tokens"], 50)
        self.assertEqual(
            summary["selected_artifact_stats"]["request_attempts"], 2
        )
        self.assertEqual(
            summary["selected_artifact_stats"]["usage"]["total_tokens"], 50
        )
        self.assertEqual(summary["config"]["api_key"], "***")

    def test_format_retry_includes_exact_validation_error(self) -> None:
        source = "First. Second."
        invalid = "I chose two steps: A0001 and A0002"
        valid = "[[COT_SPLIT_V1]]\nA0001\nA0002\n[[/COT_SPLIT_V1]]"
        client, completions = client_for([response(invalid), response(valid)])
        with tempfile.TemporaryDirectory() as directory:
            result = asyncio.run(split_cot_rows(
                [{"name": "retry", "post_think_cot": source}],
                self._config(),
                directory,
                client=client,
            ))["retry"]

        self.assertTrue(result.ok)
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(result.usage["total_tokens"], 50)
        correction = completions.calls[1]["messages"][-1]["content"]
        self.assertIn("VALIDATION ERROR:", correction)
        self.assertIn("exactly one COT_SPLIT_V1 block", correction)
        self.assertEqual(completions.calls[1]["messages"][-2], {
            "role": "assistant", "content": invalid,
        })

    def test_permanent_invalid_output_is_explicit_and_never_falls_back(self) -> None:
        client, completions = client_for([
            response("bad output"),
            response("still bad"),
        ])
        with tempfile.TemporaryDirectory() as directory:
            result = asyncio.run(split_cot_rows(
                [{"name": "bad", "post_think_cot": "First. Second."}],
                self._config(),
                directory,
                client=client,
            ))["bad"]

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "format_error")
        self.assertEqual(result.sections, [])
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(len(completions.calls), 2)

    def test_successful_cache_is_reused_without_an_llm_request(self) -> None:
        rows = [{"name": "cached", "post_think_cot": "First. Second."}]
        valid = "[[COT_SPLIT_V1]]\nA0002\n[[/COT_SPLIT_V1]]"
        with tempfile.TemporaryDirectory() as directory:
            first_client, first_calls = client_for([response(valid)])
            first = asyncio.run(split_cot_rows(
                rows, self._config(), directory, client=first_client,
            ))["cached"]
            second_client, second_calls = client_for([])
            second = asyncio.run(split_cot_rows(
                rows, self._config(), directory, client=second_client,
            ))["cached"]
            artifact_count = len(
                (Path(directory) / RESULTS_FILENAME).read_text(encoding="utf-8").splitlines()
            )
            summary = json.loads(
                (Path(directory) / SUMMARY_FILENAME).read_text(encoding="utf-8")
            )

        self.assertTrue(first.ok)
        self.assertFalse(first.cached)
        self.assertEqual(len(first_calls.calls), 1)
        self.assertTrue(second.ok)
        self.assertTrue(second.cached)
        self.assertEqual(len(second_calls.calls), 0)
        self.assertEqual(artifact_count, 1)
        self.assertEqual(summary["cached_rows"], 1)
        self.assertEqual(summary["request_attempts"], 0)
        self.assertEqual(summary["usage"], {})
        self.assertEqual(
            summary["selected_artifact_stats"]["request_attempts"], 1
        )
        self.assertEqual(
            summary["selected_artifact_stats"]["usage"]["total_tokens"], 25
        )

    def test_config_rejects_more_than_two_format_attempts(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 1 or 2"):
            self._config(max_format_attempts=3)

    def test_config_accepts_yaml_max_attempts_alias(self) -> None:
        config = LLMCotSplitterConfig.from_value({
            "model": "test-model",
            "max_attempts": 1,
        })
        self.assertEqual(config.max_format_attempts, 1)


if __name__ == "__main__":
    unittest.main()
