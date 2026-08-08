from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from blueprint import BlueprintGenerationError, phase1_request_max_tokens  # noqa: E402


class Phase1TokenBudgetTest(unittest.TestCase):
    def environment(self) -> dict[str, str]:
        return {
            "GOEDEL_PHASE1_MODEL_MAX_CONTEXT": "40960",
            "GOEDEL_PHASE1_CONTEXT_SAFETY_MARGIN": "512",
            "GOEDEL_PHASE1_MAX_OUTPUT_CAP": "16384",
            "GOEDEL_PHASE1_MIN_OUTPUT_TOKENS": "512",
            "GOEDEL_TOKENIZER_PATH": "/unused/tokenizer",
        }

    def tokenizer(self, prompt_tokens: int) -> Mock:
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = [0] * prompt_tokens
        return tokenizer

    def test_retry_uses_remaining_context_instead_of_fixed_output(self) -> None:
        messages = [{"role": "user", "content": "x"}]
        with (
            patch.dict(os.environ, self.environment(), clear=False),
            patch("blueprint._load_phase1_tokenizer", return_value=self.tokenizer(24577)),
        ):
            self.assertEqual(phase1_request_max_tokens(messages), 15871)

    def test_keeps_configured_cap_when_space_is_sufficient(self) -> None:
        messages = [{"role": "user", "content": "x"}]
        with (
            patch.dict(os.environ, self.environment(), clear=False),
            patch("blueprint._load_phase1_tokenizer", return_value=self.tokenizer(5000)),
        ):
            self.assertEqual(phase1_request_max_tokens(messages), 16384)

    def test_fails_locally_when_no_safe_output_space_remains(self) -> None:
        messages = [{"role": "user", "content": "x"}]
        with (
            patch.dict(os.environ, self.environment(), clear=False),
            patch("blueprint._load_phase1_tokenizer", return_value=self.tokenizer(40000)),
        ):
            with self.assertRaisesRegex(BlueprintGenerationError, "insufficient_context"):
                phase1_request_max_tokens(messages)


if __name__ == "__main__":
    unittest.main()
