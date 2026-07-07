"""Extract the fully-substituted Lean proof file from a phase checkpoint.

Usage: python extract_proof.py <thm_name> [output_dir]

Reads results/single_test/checkpoints/<thm_name>.json, splices every proved
node's proof body into the blueprint's `sorry_using [...]` placeholders via
the same _substitute_proof used by the pipeline, and writes the result to
<output_dir>/<thm_name>.lean (default output_dir: results/single_test/proofs).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from checkpoint import CheckpointState
from pipeline import _substitute_proof


def extract(thm_name: str, checkpoint_dir: Path, output_dir: Path) -> Path:
    ckpt_path = checkpoint_dir / f"{thm_name}.json"
    state = CheckpointState.load(ckpt_path)
    lean = state.blueprint_lean_file
    for name, body in state.proved_cache.items():
        lean = _substitute_proof(lean, name, body)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{thm_name}.lean"
    out_path.write_text(lean)
    remaining_sorry = lean.count("sorry_using")
    return out_path, remaining_sorry, state.done, state.success


if __name__ == "__main__":
    thm_name = sys.argv[1]
    base = Path(__file__).resolve().parent
    checkpoint_dir = base / "checkpoints"
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else base / "proofs"
    out_path, remaining_sorry, done, success = extract(thm_name, checkpoint_dir, output_dir)
    print(f"wrote {out_path} (remaining sorry_using: {remaining_sorry}, done={done}, success={success})")
