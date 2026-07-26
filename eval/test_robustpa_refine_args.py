from __future__ import annotations

from experiments.robustpa_refine.run_robustpa_refine import parse_args


BASE_ARGS = [
    "--config",
    "/tmp/no_such_robustpa_config.yaml",
    "--model",
    "Qwen3.5-397B-A17B-FP8",
    "--split",
    "miniF2F",
    "--subset",
    "local_number_edit_proof",
]


def test_robustpa_refine_exp_name_defaults_to_model_split_subset() -> None:
    args = parse_args(BASE_ARGS)

    assert args.exp_name == "Qwen3_5_397B_A17B_FP8_miniF2F_local_number_edit_proof"


def test_robustpa_refine_exp_name_cli_override_wins() -> None:
    args = parse_args(["--exp-name", "manual"] + BASE_ARGS)

    assert args.exp_name == "manual"
