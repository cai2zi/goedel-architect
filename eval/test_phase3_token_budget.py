from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from refinement import SemanticRefinementError, phase3_request_max_tokens  # noqa: E402


class Phase3TokenBudgetTest(unittest.TestCase):
    def environment(self) -> dict[str, str]:
        return {
            "GOEDEL_PHASE3_MODEL_MAX_CONTEXT": "40960",
            "GOEDEL_PHASE3_CONTEXT_SAFETY_MARGIN": "512",
            "GOEDEL_PHASE3_MAX_OUTPUT_CAP": "16384",
            "GOEDEL_PHASE3_MIN_OUTPUT_TOKENS": "512",
            "GOEDEL_TOKENIZER_PATH": "/unused/tokenizer",
        }

    def test_uses_remaining_context_instead_of_fixed_output(self) -> None:
        messages = [{"role": "user", "content": "x"}]
        with (
            patch.dict(os.environ, self.environment(), clear=False),
            patch("refinement._load_phase3_tokenizer", return_value=object()),
            patch("refinement._phase3_message_token_count", return_value=24577),
        ):
            self.assertEqual(phase3_request_max_tokens(messages), 15871)

    def test_fails_locally_when_no_safe_output_space_remains(self) -> None:
        messages = [{"role": "user", "content": "x"}]
        with (
            patch.dict(os.environ, self.environment(), clear=False),
            patch("refinement._load_phase3_tokenizer", return_value=object()),
            patch("refinement._phase3_message_token_count", return_value=40000),
        ):
            with self.assertRaisesRegex(SemanticRefinementError, "insufficient_context"):
                phase3_request_max_tokens(messages)


if __name__ == "__main__":
    unittest.main()
