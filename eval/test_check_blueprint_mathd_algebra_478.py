"""Smoke test for LeanCompiler.check_blueprint on mathd_algebra_478.

Run from the repository root:

    python eval/test_check_blueprint_mathd_algebra_478.py

This is an integration test: it calls `lake env lean` through
src/lean_compiler.py, so `lake` must be on PATH and `goedel_lean/` must have
its dependencies built.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_LEAN_PROJECT_ROOT = REPO_ROOT / "goedel_lean"

sys.path.insert(0, str(SRC_ROOT))

from lean_compiler import LeanCompiler  # noqa: E402


TARGET_NAME = "mathd_algebra_478"

LEAN_CODE = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The volume of a cone is given by the formula $V = \\frac{1}{3}Bh$, where $B$ is the area of the base and $h$ is the height. The area of the base of a cone is 30 square units, and its height is 6.5 units. What is the number of cubic units in its volume? Show that it is 65.-/
@[blueprint (statement := /-- The volume of a cone is given by the formula V = 1/3 Bh, where B is the area of the base and h is the height. The area of the base of a cone is 30 square units, and its height is 6.5 units. What is the number of cubic units in its volume? Show that it is 65.-/)
 (proof := /-- Substitute h₂ and h₃ into h₁ and compute: v = 1/3*(30*(13/2)) = 1/3*195 = 65. -/)]
theorem mathd_algebra_478 (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) (h₁ : v = 1 / 3 * (b * h))
 (h₂ : b = 30) (h₃ : h = 13 / 2) : v = 65 := by
 sorry_using []"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real check_blueprint smoke test for mathd_algebra_478."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_LEAN_PROJECT_ROOT,
        help="Lean/Lake project root to use for `lake env lean`.",
    )
    parser.add_argument(
        "--target",
        default=TARGET_NAME,
        help="The theorem name passed to check_blueprint.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print the full raw Lean output instead of only the tail on failure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if shutil.which("lake") is None:
        print("ERROR: `lake` is not on PATH, so check_blueprint cannot run.")
        print("Install elan/lake or run this from an environment where `lake` is available.")
        return 2

    compiler = LeanCompiler(project_root=args.project_root)
    result = compiler.check_blueprint(LEAN_CODE, args.target)

    print(f"target: {args.target}")
    print(f"project_root: {args.project_root}")
    print(f"success: {result.success}")
    print(f"validated: {result.validated}")

    if result.warnings:
        print("\nwarnings:")
        for warning in result.warnings:
            print(f"- {warning}")

    if result.errors:
        print("\nerrors:")
        for error in result.errors:
            print(f"- {error}")

    if not result.success:
        raw = result.raw_output if args.show_raw else result.raw_output[-4000:]
        if raw:
            print("\nraw Lean output:")
            print(raw)
        return 1

    print("\ncheck_blueprint OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
