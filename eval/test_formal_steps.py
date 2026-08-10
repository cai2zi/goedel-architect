from __future__ import annotations

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
    merge_nonsemantic_spans,
    parse_boundaries,
    parse_split_response,
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

    def test_manifest_treats_final_restatement_as_an_ordinary_step(self) -> None:
        source = "Derive x = 15.\nTherefore, the answer is \\boxed{15}.\n"
        boundary = source.index("Therefore")
        manifest = make_formal_step_manifest(source, [(0, boundary), (boundary, len(source))])
        steps = manifest["steps"]
        self.assertEqual("".join(step["source_text"] for step in steps), source)
        self.assertEqual(set(steps[1]), {
            "step_id", "source_start", "source_end", "source_text", "source_sha256",
        })

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
        content = "[[FORMAL_STEPS_V4]]\n" + "\n".join([
            anchors[0]["anchor_id"], anchors[-1]["anchor_id"],
        ]) + "\n[[/FORMAL_STEPS_V4]]"
        boundaries = parse_boundaries(content, anchors)
        spans = spans_from_boundaries(source, anchors, boundaries)
        self.assertEqual("".join(source[a:b] for a, b in spans), source)

    def test_boundary_parser_tolerates_single_bracket_closing_marker(self) -> None:
        source = "First inference. Therefore second inference.\n"
        anchors = make_boundary_anchors(source)
        final = anchors[-1]["anchor_id"]
        self.assertEqual(
            parse_boundaries(
                f"[[FORMAL_STEPS_V4]]\n{final}\n[/FORMAL_STEPS_V4]", anchors
            ),
            [final],
        )

    def test_heading_and_colon_cannot_end_a_step(self) -> None:
        source = "### Compute\nNow calculate:\n$x=1$.\n"
        anchors = make_boundary_anchors(source)
        for forbidden in anchors[:-1]:
            if forbidden["kind"] == "heading" or forbidden["source_text"].rstrip().endswith(":"):
                content = (
                    "[[FORMAL_STEPS_V4]]\n" + forbidden["anchor_id"] + "\n"
                    + anchors[-1]["anchor_id"] + "\n[[/FORMAL_STEPS_V4]]"
                )
                with self.assertRaises(FormalStepSplitError):
                    parse_boundaries(content, anchors)

    def test_parser_reports_all_forbidden_boundaries_in_one_attempt(self) -> None:
        source = (
            "First calculation:\n"
            "$x=1$.\n"
            "### Next calculation\n"
            "Now calculate:\n"
            "$y=2$.\n"
        )
        anchors = make_boundary_anchors(source)
        forbidden = [
            anchor["anchor_id"]
            for anchor in anchors[:-1]
            if anchor["kind"] == "heading"
            or anchor["source_text"].rstrip().endswith(":")
        ]
        content = (
            "[[FORMAL_STEPS_V4]]\n"
            + "\n".join([*forbidden, anchors[-1]["anchor_id"]])
            + "\n[[/FORMAL_STEPS_V4]]"
        )
        with self.assertRaises(FormalStepSplitError) as caught:
            parse_split_response(content, anchors)
        reported = {
            boundary
            for boundaries in caught.exception.forbidden_boundaries.values()
            for boundary in boundaries
        }
        self.assertEqual(reported, set(forbidden))
        for boundary in forbidden:
            self.assertIn(boundary, caught.exception.reason)

    def test_prompt_has_soft_step_prior(self) -> None:
        source = "We compute x. Therefore x=1. " * 30
        anchors = make_boundary_anchors(source)
        config = FormalStepSplitterConfig(model="m", target_tokens_per_step=100)
        prompt = build_split_messages(source, anchors, config)[0]["content"]
        self.assertIn("soft prior", prompt)
        self.assertIn("not a quota", prompt)
        self.assertIn("ordinary source content", prompt)
        self.assertNotIn("OMIT_FINAL_RESTATEMENT", prompt)
        self.assertNotIn("COT_CLAIM", prompt)

    def test_postprocess_merges_format_only_span_losslessly(self) -> None:
        source = "Compute x = 7.\n$$\nTherefore x is known.\n"
        first_end = source.index("$$")
        format_end = first_end + 3
        spans, formatting = merge_nonsemantic_spans(
            source, [(0, first_end), (first_end, format_end), (format_end, len(source))],
        )
        self.assertEqual(formatting, 1)
        self.assertEqual("".join(source[a:b] for a, b in spans), source)

    def test_splitter_keeps_repeated_boxed_final_answer(self) -> None:
        source = "Solving gives x = 115.\nTherefore, the final answer is $\\boxed{115}$.\n"
        boundary = source.index("Therefore")
        anchors = make_boundary_anchors(source)
        content = (
            "[[FORMAL_STEPS_V4]]\n"
            f"{anchors[0]['anchor_id']}\n"
            f"{anchors[-1]['anchor_id']}\n"
            "[[/FORMAL_STEPS_V4]]"
        )
        boundaries = parse_split_response(content, anchors)
        spans, formatting = merge_nonsemantic_spans(
            source, [(0, boundary), (boundary, len(source))],
        )
        self.assertEqual(formatting, 0)
        self.assertEqual(len(spans), 2)
        self.assertEqual(boundaries[-1], anchors[-1]["anchor_id"])

    def test_old_omit_marker_is_rejected(self) -> None:
        source = "First. Second. Final."
        anchors = make_boundary_anchors(source)
        content = (
            "[[FORMAL_STEPS_V4]]\n"
            f"{anchors[0]['anchor_id']} | OMIT_FINAL_RESTATEMENT\n"
            f"{anchors[-1]['anchor_id']}\n"
            "[[/FORMAL_STEPS_V4]]"
        )
        with self.assertRaises(FormalStepSplitError):
            parse_split_response(content, anchors)

    def test_postprocess_keeps_new_final_inference(self) -> None:
        source = "We know x + y = 5.\nTherefore, the final answer is $\\boxed{5}$.\n"
        boundary = source.index("Therefore")
        spans, _formatting = merge_nonsemantic_spans(
            source, [(0, boundary), (boundary, len(source))],
        )
        self.assertEqual(len(spans), 2)


if __name__ == "__main__":
    unittest.main()
