from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from cot_blueprint_refine.common import read_jsonl  # noqa: E402
from cot_blueprint_refine.cot_steps import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    decode_steps,
    encode_steps,
    split_cot_steps,
)
from cot_blueprint_refine.llm_cot_splitter import (  # noqa: E402
    SplitResult,
    atomize_cot,
    sections_from_boundaries,
)
from cot_blueprint_refine.prepare_inputs import (  # noqa: E402
    DATASET_SUBSET,
    write_generation_artifacts,
)
from cot_blueprint_refine.run_cot_split import run_cot_split  # noqa: E402


def generation_row(source_text: str) -> dict:
    return {
        "name": "demo/1",
        "source": "demo",
        "row_index": 0,
        "problem": "What is x?",
        "claimed_answer": "2",
        "post_think_cot": source_text,
        "informal_statement": "statement",
        "informal_proof": source_text,
        "cot_manifest_json": "old-manifest",
    }


class CotSplitStageTest(unittest.TestCase):
    def config(self, root: Path, mode: str):
        return OmegaConf.create({
            "output_base": str(root),
            "exp_name": "unit",
            "cot_splitter": {
                "mode": mode,
                "model": "model",
                "openai_base_url": "http://localhost:8001/v1",
                "api_key": "dummy",
                "concurrency": 2,
                "temperature": 0,
                "max_tokens": 128,
                "timeout_s": 30,
                "max_attempts": 2,
                "enable_thinking": False,
                "fallback_on_error": False,
            },
        })

    def test_deterministic_mode_validates_without_rewriting_artifacts(self) -> None:
        source = "### Step 1: Compute\nx = 1.\n\n### Final Answer\n\\boxed{1}"
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), "deterministic")
            original = generation_row(source)
            original["cot_manifest_json"] = encode_steps(split_cot_steps(source))
            write_generation_artifacts(config, [original])
            before = read_jsonl(
                Path(temporary) / "unit" / "prepared" / "generation_inputs.jsonl"
            )[0]

            metrics = run_cot_split(config)

            root = Path(temporary) / "unit" / "prepared"
            json_row = read_jsonl(root / "generation_inputs.jsonl")[0]
            parquet_path = next((root / "data" / DATASET_SUBSET).glob("*.parquet"))
            parquet_row = pq.read_table(parquet_path).to_pylist()[0]
            self.assertEqual(json_row["cot_manifest_json"], parquet_row["cot_manifest_json"])
            self.assertEqual(json_row, before)
            self.assertNotIn("cot_splitter_mode", json_row)
            self.assertGreater(len(decode_steps(json_row["cot_manifest_json"])), 0)
            self.assertEqual(metrics["rows"], 1)

    def test_llm_mode_installs_exact_single_claim_steps(self) -> None:
        source = "### Setup\nLet x = 1.\n\n### Conclusion\nTherefore x + 1 = 2."
        atoms = atomize_cot(source)
        boundaries = [atoms[1]["atom_id"], atoms[-1]["atom_id"]]
        sections = sections_from_boundaries(source, atoms, boundaries)
        result = SplitResult(
            row_id="demo/1",
            status="ok",
            source_sha256="source-hash",
            cache_key="cache-key",
            atoms=atoms,
            boundaries=boundaries,
            sections=sections,
            attempts=[{"attempt": 1}],
        )

        async def fake_split(*_args, **_kwargs):
            return {"demo/1": result}

        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), "llm_boundary_v1")
            write_generation_artifacts(config, [generation_row(source)])
            with patch(
                "cot_blueprint_refine.run_cot_split.split_cot_rows",
                side_effect=fake_split,
            ):
                metrics = run_cot_split(config)

            row = read_jsonl(
                Path(temporary) / "unit" / "prepared" / "generation_inputs.jsonl"
            )[0]
            steps = decode_steps(row["cot_manifest_json"])
            self.assertEqual("".join(step["source_text"] for step in steps), source)
            self.assertTrue(all(len(step["claims"]) == 1 for step in steps))
            self.assertTrue(all(step.get("segments") for step in steps))
            self.assertEqual(row["cot_splitter_fallback"], False)
            self.assertEqual(metrics["fallback_rows"], 0)

    def test_llm_failure_does_not_replace_prepared_manifest(self) -> None:
        source = "A wrong claim. Therefore \\boxed{3}."
        failed = SplitResult(
            row_id="demo/1",
            status="format_error",
            source_sha256="source-hash",
            cache_key="cache-key",
            error="bad boundary",
        )

        async def fake_split(*_args, **_kwargs):
            return {"demo/1": failed}

        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), "llm_boundary_v1")
            original = generation_row(source)
            write_generation_artifacts(config, [original])
            with patch(
                "cot_blueprint_refine.run_cot_split.split_cot_rows",
                side_effect=fake_split,
            ):
                with self.assertRaisesRegex(RuntimeError, "valid lossless partition"):
                    run_cot_split(config)

            retained = read_jsonl(
                Path(temporary) / "unit" / "prepared" / "generation_inputs.jsonl"
            )[0]
            self.assertEqual(retained["cot_manifest_json"], "old-manifest")

    def test_llm_mode_builds_structured_subclaims_and_shape_metrics(self) -> None:
        source = (
            "### Cases\n"
            "- First, x = 1.\n"
            "- Second, y = 2.\n"
            "| item | value |\n"
            "|---|---|\n"
            "| x | 1 |\n"
            "| y | 2 |"
        )
        atoms = atomize_cot(source)
        boundaries = [atoms[-1]["atom_id"]]
        result = SplitResult(
            row_id="demo/1",
            status="ok",
            source_sha256="source-hash",
            cache_key="cache-key",
            prompt_content_sha256="prompt-hash",
            atoms=atoms,
            boundaries=boundaries,
            sections=sections_from_boundaries(source, atoms, boundaries),
            attempts=[{"attempt": 1}],
        )

        async def fake_split(*_args, **_kwargs):
            return {"demo/1": result}

        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), "llm_boundary_v4")
            write_generation_artifacts(config, [generation_row(source)])
            with patch(
                "cot_blueprint_refine.run_cot_split.split_cot_rows",
                side_effect=fake_split,
            ):
                metrics = run_cot_split(config)
            row = read_jsonl(
                Path(temporary) / "unit" / "prepared" / "generation_inputs.jsonl"
            )[0]

        step = decode_steps(row["cot_manifest_json"], source=source)[0]
        self.assertEqual(len(step["claims"]), 4)
        self.assertEqual(
            [claim["claim_kind"] for claim in step["claims"]],
            ["list_item", "list_item", "table_data_row", "table_data_row"],
        )
        self.assertEqual(metrics["claims_per_step"]["max"], 4)
        self.assertEqual(metrics["steps_with_layout_context"], 1)
        self.assertEqual(metrics["max_atoms_per_step"], len(atoms))
        self.assertEqual(row["cot_manifest_schema_version"], MANIFEST_SCHEMA_VERSION)
        self.assertEqual(row["cot_splitter_prompt_content_sha256"], "prompt-hash")


if __name__ == "__main__":
    unittest.main()
