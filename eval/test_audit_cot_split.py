from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from cot_blueprint_refine.audit_cot_split import (  # noqa: E402
    _is_pure_title,
    audit_dataset,
    render_table,
)
from cot_blueprint_refine.llm_cot_splitter import atomize_cot  # noqa: E402


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CotSplitAuditTest(unittest.TestCase):
    def test_bold_list_assertion_is_not_misclassified_as_title(self) -> None:
        self.assertTrue(_is_pure_title("- **Area**:"))
        self.assertFalse(_is_pure_title("- **AB = BC**"))

    def test_reports_lossless_layout_subclaims_and_compound_steps(self) -> None:
        source = """### Context Only
### Compute
Values:
- $x = 1$
- $y = 2$

### Final Answer
$$
x + y = 3
$$"""
        atoms = atomize_cot(source)
        self.assertEqual([atom["kind"] for atom in atoms], [
            "heading", "heading", "prose", "list_item", "list_item",
            "heading", "display_math",
        ])

        groups = [atoms[:1], atoms[1:5], atoms[5:]]
        steps = []
        for index, group in enumerate(groups, start=1):
            start = group[0]["source_start"]
            end = group[-1]["source_end"]
            text = source[start:end]
            context = index == 1
            claims = [] if context else [{
                "claim_id": f"S{index:03d}.C001",
                "source_text": text,
                "source_start": 0,
                "source_end": len(text),
                "source_sha256": sha256(text),
            }]
            steps.append({
                "step_id": f"S{index:03d}",
                "source_text": text,
                "source_start": start,
                "source_end": end,
                "source_sha256": sha256(text),
                "atom_ids": [atom["atom_id"] for atom in group],
                "role": "context" if context else "derived_claim",
                "requires_formalization": not context,
                "claims": claims,
            })

        generation_row = {
            "name": "demo/1",
            "source": "demo",
            "row_index": 0,
            "post_think_cot": source,
            "cot_manifest_json": json.dumps(steps),
            "cot_splitter_version": "llm-cot-boundary-v4",
        }
        split_row = {
            "row_id": "demo/1",
            "status": "ok",
            "source_sha256": sha256(source),
            "splitter_version": "llm-cot-boundary-v4",
            "atoms": atoms,
        }

        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary)
            (prepared / "cot_splitter").mkdir()
            (prepared / "generation_inputs.jsonl").write_text(
                json.dumps(generation_row) + "\n", encoding="utf-8"
            )
            (prepared / "cot_splitter" / "llm_cot_splits.jsonl").write_text(
                json.dumps(split_row) + "\n", encoding="utf-8"
            )

            report = audit_dataset(prepared, compound_threshold=2)

        summary = report["summary"]
        row = report["rows"][0]
        self.assertEqual(summary["lossless_row_count"], 1)
        self.assertEqual(summary["atom_lossless_row_count"], 1)
        self.assertEqual(row["context_only_pure_title_step_count"], 1)
        self.assertEqual(row["required_pure_title_step_count"], 0)
        self.assertEqual(row["structural_claim_coverage"]["list_item"]["total"], 2)
        self.assertEqual(
            row["structural_claim_coverage"]["list_item"]["dedicated_claim"], 0
        )
        self.assertEqual(
            row["structural_claim_coverage"]["display_math"]["dedicated_claim"], 1
        )
        self.assertEqual(row["compound_step_count"], 1)
        self.assertIn("demo/1", render_table(report))


if __name__ == "__main__":
    unittest.main()
