from __future__ import annotations

import copy
import hashlib
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from cot_blueprint_refine.cot_manifest_validation import (  # noqa: E402
    OFFSET_SPACE,
    SplitManifestValidationError,
    validate_split_manifest,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _structured_step(
    pieces: list[tuple[str, str]],
    *,
    step_number: int = 1,
    global_start: int = 0,
    first_atom_number: int = 1,
) -> dict:
    step_id = f"S{step_number:03d}"
    source_text = "".join(text for _kind, text in pieces)
    atoms: list[dict] = []
    claims: list[dict] = []
    segments: list[dict] = []
    cursor = 0
    for piece_index, (kind, text) in enumerate(pieces):
        start = cursor
        end = start + len(text)
        atom_id = f"A{first_atom_number + piece_index:04d}"
        atoms.append({
            "atom_id": atom_id,
            "kind": "prose",
            "source_start": start,
            "source_end": end,
            "source_text": text,
            "source_sha256": _sha256(text),
        })
        segment = {
            "kind": kind,
            "source_start": start,
            "source_end": end,
            "source_text": text,
            "source_sha256": _sha256(text),
        }
        if kind == "claim":
            claim_id = f"{step_id}.C{len(claims) + 1:03d}"
            claims.append({
                "claim_id": claim_id,
                "source_start": start,
                "source_end": end,
                "source_text": text,
                "source_sha256": _sha256(text),
                "atom_ids": [atom_id],
            })
            segment["claim_id"] = claim_id
        segments.append(segment)
        cursor = end
    return {
        "step_id": step_id,
        "source_start": global_start,
        "source_end": global_start + len(source_text),
        "source_text": source_text,
        "source_sha256": _sha256(source_text),
        "offset_space": dict(OFFSET_SPACE),
        "atoms": atoms,
        "atom_ids": [atom["atom_id"] for atom in atoms],
        "claims": claims,
        "segments": segments,
        "requires_formalization": bool(claims),
    }


def _duplicate_text_fixture() -> tuple[str, list[dict]]:
    pieces = [
        ("context", "Intro: "),
        ("claim", "x = 1."),
        ("context", " Again, "),
        ("claim", "x = 1."),
    ]
    source = "".join(text for _kind, text in pieces)
    return source, [_structured_step(pieces)]


def _two_step_fixture() -> tuple[str, list[dict]]:
    first = "First claim. "
    second = "Second claim."
    return first + second, [
        _structured_step(
            [("claim", first)],
            step_number=1,
            global_start=0,
            first_atom_number=1,
        ),
        _structured_step(
            [("claim", second)],
            step_number=2,
            global_start=len(first),
            first_atom_number=2,
        ),
    ]


class SplitManifestValidationTest(unittest.TestCase):
    def assert_invalid(self, source: str, steps: list[dict], pattern: str) -> None:
        with self.assertRaisesRegex(SplitManifestValidationError, pattern):
            validate_split_manifest(source, steps)

    def test_duplicate_equal_claim_text_and_hash_are_unambiguous(self) -> None:
        source, steps = _duplicate_text_fixture()

        validate_split_manifest(source, steps)

        claims = steps[0]["claims"]
        self.assertEqual(claims[0]["source_text"], claims[1]["source_text"])
        self.assertEqual(claims[0]["source_sha256"], claims[1]["source_sha256"])
        self.assertNotEqual(claims[0]["claim_id"], claims[1]["claim_id"])
        self.assertNotEqual(claims[0]["source_start"], claims[1]["source_start"])

    def test_global_step_spans_cover_only_the_stripped_source(self) -> None:
        body = "First claim."
        source = f" \n{body}\t "
        start = source.index(body)
        steps = [_structured_step(
            [("claim", body)],
            global_start=start,
        )]

        validate_split_manifest(source, steps)

    def test_rejects_global_step_gap_and_overlap(self) -> None:
        source, original = _two_step_fixture()
        for label, changed_start, pattern in (
            ("gap", original[1]["source_start"] + 1, "step gap"),
            ("overlap", original[1]["source_start"] - 1, "step overlap"),
        ):
            with self.subTest(label=label):
                steps = copy.deepcopy(original)
                steps[1]["source_start"] = changed_start
                self.assert_invalid(source, steps, pattern)

    def test_rejects_atom_gap_and_overlap(self) -> None:
        source, original = _duplicate_text_fixture()
        expected = original[0]["atoms"][0]["source_end"]
        for label, changed_start, pattern in (
            ("gap", expected + 1, "atom gap"),
            ("overlap", expected - 1, "atom overlap"),
        ):
            with self.subTest(label=label):
                steps = copy.deepcopy(original)
                steps[0]["atoms"][1]["source_start"] = changed_start
                self.assert_invalid(source, steps, pattern)

    def test_rejects_segment_gap_and_overlap(self) -> None:
        source, original = _duplicate_text_fixture()
        expected = original[0]["segments"][0]["source_end"]
        for label, changed_start, pattern in (
            ("gap", expected + 1, "segment gap"),
            ("overlap", expected - 1, "segment overlap"),
        ):
            with self.subTest(label=label):
                steps = copy.deepcopy(original)
                steps[0]["segments"][1]["source_start"] = changed_start
                self.assert_invalid(source, steps, pattern)

    def test_rejects_unknown_claim_binding(self) -> None:
        source, steps = _duplicate_text_fixture()
        steps[0]["segments"][1]["claim_id"] = "S001.C999"

        self.assert_invalid(source, steps, "unknown claim ID S001.C999")

    def test_rejects_missing_or_reused_claim_segment(self) -> None:
        source, original = _duplicate_text_fixture()

        missing = copy.deepcopy(original)
        missing[0]["segments"][3]["kind"] = "context"
        del missing[0]["segments"][3]["claim_id"]
        self.assert_invalid(source, missing, "claims missing a unique segment: S001.C002")

        reused = copy.deepcopy(original)
        reused[0]["segments"][3]["claim_id"] = "S001.C001"
        self.assert_invalid(source, reused, "represented by more than one segment")

    def test_rejects_hash_mismatches_at_every_identity_level(self) -> None:
        source, original = _duplicate_text_fixture()
        locations = (
            ("step", lambda steps: steps[0]),
            ("atom", lambda steps: steps[0]["atoms"][0]),
            ("claim", lambda steps: steps[0]["claims"][0]),
            ("segment", lambda steps: steps[0]["segments"][0]),
        )
        for label, select in locations:
            with self.subTest(label=label):
                steps = copy.deepcopy(original)
                select(steps)["source_sha256"] = "0" * 64
                self.assert_invalid(source, steps, "hash mismatch")

    def test_rejects_noncanonical_id_order(self) -> None:
        source, original = _duplicate_text_fixture()
        cases = (
            ("step", lambda steps: steps[0].__setitem__("step_id", "S002"), "expected S001"),
            (
                "claim",
                lambda steps: steps[0]["claims"][0].__setitem__("claim_id", "S001.C002"),
                "expected S001.C001",
            ),
            (
                "atom",
                lambda steps: steps[0]["atoms"][1].__setitem__("atom_id", "A0001"),
                "duplicate atom ID A0001",
            ),
        )
        for label, mutate, pattern in cases:
            with self.subTest(label=label):
                steps = copy.deepcopy(original)
                mutate(steps)
                self.assert_invalid(source, steps, pattern)

    def test_rejects_claim_text_span_mismatch(self) -> None:
        source, steps = _duplicate_text_fixture()
        steps[0]["claims"][0]["source_text"] = "x = 2."
        steps[0]["claims"][0]["source_sha256"] = _sha256("x = 2.")

        self.assert_invalid(source, steps, "does not equal the claim span")

    def test_requires_formalization_is_exactly_equivalent_to_claim_presence(self) -> None:
        source, original = _duplicate_text_fixture()
        steps = copy.deepcopy(original)
        steps[0]["requires_formalization"] = False
        self.assert_invalid(source, steps, "must be true exactly when")

        context_source = "Narration only."
        context_steps = [_structured_step([("context", context_source)])]
        validate_split_manifest(context_source, context_steps)
        context_steps[0]["requires_formalization"] = True
        self.assert_invalid(context_source, context_steps, "must be true exactly when")

    def test_rejects_wrong_offset_space_declaration(self) -> None:
        source, steps = _duplicate_text_fixture()
        steps[0]["offset_space"]["claims"] = "global"

        self.assert_invalid(source, steps, "offset_space.*must equal")

    def test_validates_scope_bindings_in_both_directions(self) -> None:
        source, steps = _duplicate_text_fixture()
        step = steps[0]
        scope = step["segments"][0]
        scope.update({
            "context_type": "narration",
            "scope_type": "claim_prefix",
            "scope_id": "S001.G001",
            "applies_to_claim_ids": ["S001.C001", "S001.C002"],
        })
        step["claims"][0]["scope_ids"] = ["S001.G001"]
        step["claims"][1]["scope_ids"] = ["S001.G001"]
        step["scope_count"] = 1

        validate_split_manifest(source, steps)

        missing_backlink = copy.deepcopy(steps)
        missing_backlink[0]["claims"][1]["scope_ids"] = []
        self.assert_invalid(source, missing_backlink, "scope_ids.*must equal")

        unknown_target = copy.deepcopy(steps)
        unknown_target[0]["segments"][0]["applies_to_claim_ids"] = ["S001.C999"]
        self.assert_invalid(source, unknown_target, "unknown claim ID")


if __name__ == "__main__":
    unittest.main()
