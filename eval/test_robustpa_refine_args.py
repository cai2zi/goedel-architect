from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from robustpa_refine.run_robustpa_refine import _validate_args  # noqa: E402


class RobustPAConfigTest(unittest.TestCase):
    def test_base_config_is_kimina_only_with_new_tool_budget(self) -> None:
        config = OmegaConf.load(
            REPO_ROOT / "experiments/robustpa_refine/configs/base.yaml"
        )
        self.assertNotIn("lean_backend", config)
        self.assertNotIn("parallel_tool_calls", config)
        self.assertNotIn("node_max_negation_probe_turns", config)
        self.assertEqual(config.max_tool_calls_per_turn, 3)

    def test_tool_budget_must_be_positive(self) -> None:
        config = OmegaConf.load(
            REPO_ROOT / "experiments/robustpa_refine/configs/base.yaml"
        )
        config.max_tool_calls_per_turn = 0
        with self.assertRaisesRegex(ValueError, "max_tool_calls_per_turn"):
            _validate_args(SimpleNamespace(**OmegaConf.to_container(config, resolve=False)))


if __name__ == "__main__":
    unittest.main()
