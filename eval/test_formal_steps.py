from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

from cot_blueprint_refine.formal_step_splitter import (  # noqa: E402
    FormalStepSplitError,
    FormalStepSplitterConfig,
    build_split_messages,
    make_boundary_anchors,
    parse_boundaries,
    spans_from_boundaries,
)
from cot_blueprint_refine.formal_steps import (  # noqa: E402
    FormalStepValidationError,
    decode_formal_step_manifest,
    encode_formal_step_manifest,
    make_formal_step_manifest,
)


class FormalStepsTest(unittest.TestCase):
    def test_manifest_is_exact_contiguous_partition(self) -> None:
        source = "Heading:\n\n$x=1$. Therefore, $y=2$.\n"
        manifest = make_formal_step_manifest(source, [(0, 10), (10, len(source))])
        decoded = decode_formal_step_manifest(encode_formal_step_manifest(manifest), source=source)
        self.assertEqual("".join(step["source_text"] for step in decoded["steps"]), source)
        self.assertEqual([step["step_id"] for step in decoded["steps"]], ["S001", "S002"])

    def test_manifest_rejects_gap(self) -> None:
        with self.assertRaises(FormalStepValidationError):
            make_formal_step_manifest("abcdef", [(0, 2), (3, 6)])

    def test_anchors_are_transport_only_and_lossless(self) -> None:
        source = "### Step 1\nWe compute:\n$$x=1.$$\nTherefore, y=2.\n"
        anchors = make_boundary_anchors(source)
        self.assertEqual("".join(item["source_text"] for item in anchors), source)
        self.assertTrue(all(set(item) >= {
            "anchor_id", "source_start", "source_end", "source_text", "source_sha256", "kind"
        } for item in anchors))

    def test_boundary_parser_and_reconstruction(self) -> None:
        source = "Let x=1.\nTherefore y=2.\nFinally z=3.\n"
        anchors = make_boundary_anchors(source)
        content = "[[FORMAL_STEPS_V1]]\n" + "\n".join([
            anchors[0]["anchor_id"], anchors[-1]["anchor_id"],
        ]) + "\n[[/FORMAL_STEPS_V1]]"
        boundaries = parse_boundaries(content, anchors)
        spans = spans_from_boundaries(source, anchors, boundaries)
        self.assertEqual("".join(source[a:b] for a, b in spans), source)

    def test_boundary_parser_tolerates_single_bracket_closing_marker(self) -> None:
        source = "First inference. Therefore second inference.\n"
        anchors = make_boundary_anchors(source)
        final = anchors[-1]["anchor_id"]
        self.assertEqual(
            parse_boundaries(
                f"[[FORMAL_STEPS_V1]]\n{final}\n[/FORMAL_STEPS_V1]", anchors
            ),
            [final],
        )

    def test_heading_and_colon_cannot_end_a_step(self) -> None:
        source = "### Compute\nNow calculate:\n$x=1$.\n"
        anchors = make_boundary_anchors(source)
        for forbidden in anchors[:-1]:
            if forbidden["kind"] == "heading" or forbidden["source_text"].rstrip().endswith(":"):
                content = (
                    "[[FORMAL_STEPS_V1]]\n" + forbidden["anchor_id"] + "\n"
                    + anchors[-1]["anchor_id"] + "\n[[/FORMAL_STEPS_V1]]"
                )
                with self.assertRaises(FormalStepSplitError):
                    parse_boundaries(content, anchors)

    def test_prompt_has_soft_step_prior(self) -> None:
        source = "We compute x. Therefore x=1. " * 30
        anchors = make_boundary_anchors(source)
        config = FormalStepSplitterConfig(model="m", target_tokens_per_step=100)
        prompt = build_split_messages(source, anchors, config)[0]["content"]
        self.assertIn("soft prior", prompt)
        self.assertIn("not a quota", prompt)
        self.assertNotIn("COT_CLAIM", prompt)


if __name__ == "__main__":
    unittest.main()
