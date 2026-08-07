from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from cot_blueprint_refine.kimina_runtime import PersistentKiminaRuntime  # noqa: E402


class FakeProcess:
    last_kwargs = None

    def __init__(self, *_args, **kwargs) -> None:
        type(self).last_kwargs = kwargs
        self.pid = 4321
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def config(root: Path):
    server_root = root / "server"
    project = root / "project"
    repl = root / "repl"
    for path in (server_root, project):
        path.mkdir()
    repl.write_text("binary", encoding="utf-8")
    return OmegaConf.create({
        "output_base": str(root / "output"),
        "exp_name": "unit",
        "python_bin": sys.executable,
        "kimina": {
            "auto_start": True, "auto_destroy": True,
            "root": str(server_root), "host": "127.0.0.1", "port": 18000,
            "startup_timeout_s": 1, "shutdown_timeout_s": 1,
            "poll_interval_s": 0.01, "metrics_interval_s": 60,
            "max_repls": 48, "max_repl_uses": 256, "max_repl_mem": "64G",
            "max_snippets_per_request": 8,
            "project_dir": str(project), "repl_path": str(repl),
        },
    })


class PersistentKiminaRuntimeTest(unittest.TestCase):
    def test_owned_process_gets_exact_environment_and_is_destroyed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = PersistentKiminaRuntime(config(Path(temporary)))
            with (
                patch.object(runtime, "_port_in_use", return_value=False),
                patch.object(runtime, "_wait_ready", side_effect=lambda: setattr(runtime, "ready_at", "ready")),
                patch("cot_blueprint_refine.kimina_runtime.subprocess.Popen", FakeProcess),
                patch("cot_blueprint_refine.kimina_runtime.os.getpgid", return_value=4321),
                patch("cot_blueprint_refine.kimina_runtime.os.killpg") as killpg,
            ):
                runtime.ensure("blueprint")
                runtime.stop()
            env = FakeProcess.last_kwargs["env"]
            self.assertEqual(env["LEAN_SERVER_MAX_REPL_MEM"], "64G")
            self.assertEqual(env["LEAN_SERVER_MAX_REPLS"], "48")
            self.assertEqual(env["LEAN_SERVER_MAX_REPL_USES"], "256")
            self.assertEqual(env["LEAN_SERVER_MAX_SNIPPETS_PER_REQUEST"], "8")
            killpg.assert_called_once_with(4321, 15)
            self.assertIn('"status": "stopped"', runtime.session_path.read_text())

    def test_refuses_to_attach_to_an_existing_port(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = PersistentKiminaRuntime(config(Path(temporary)))
            with patch.object(runtime, "_port_in_use", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "already in use"):
                    runtime.start()


if __name__ == "__main__":
    unittest.main()
