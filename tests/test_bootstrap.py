import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from bootstrap import run_bootstrap


class FakeRunner:
    def __init__(self, python_version="Python 3.12.14"):
        self.commands = []
        self.python_version = python_version

    def __call__(self, command, **kwargs):
        command = [str(part) for part in command]
        self.commands.append(command)
        cwd = Path(kwargs["cwd"])
        if (command[0] == "uv" and command[1:3] == ["venv", "--python"]) or command[-3:] == ["-m", "venv", ".venv"]:
            venv_python = cwd / ".venv" / "Scripts" / "python.exe"
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.touch()
        if command[-1:] == ["--version"]:
            return subprocess.CompletedProcess(command, 0, stdout=self.python_version, stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


class BootstrapTests(unittest.TestCase):
    def test_setup_only_prompts_securely_and_preserves_existing_env_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env"
            env_file.write_text(
                "LLM_PROVIDER=anthropic\nOPENAI_API_KEY=keep-this-key\nCUSTOM_SETTING=keep-this-value\n",
                encoding="utf-8",
            )
            runner = FakeRunner()
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                result = run_bootstrap(
                    ["--setup-only"],
                    root=root,
                    runner=runner,
                    uv_path="uv",
                    input_fn=lambda prompt: self.fail("Existing provider should be reused"),
                    getpass_fn=lambda prompt: "anthropic-test-secret",
                    environ={},
                    interactive=True,
                )
            updated = env_file.read_text(encoding="utf-8")
            self.assertEqual(result, 0)
            self.assertIn("OPENAI_API_KEY=keep-this-key", updated)
            self.assertIn("CUSTOM_SETTING=keep-this-value", updated)
            self.assertIn('ANTHROPIC_API_KEY="anthropic-test-secret"', updated)
            self.assertNotIn("anthropic-test-secret", output.getvalue())
            self.assertFalse(any("main.py" in command for command in runner.commands))
            self.assertTrue(any(command[:2] == ["uv", "python"] for command in runner.commands))
            self.assertTrue(any(command[:2] == ["uv", "pip"] for command in runner.commands))

    def test_noninteractive_missing_key_fails_helpfully_without_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".env").write_text(
                "LLM_PROVIDER= openai # saved provider\nOPENAI_API_KEY= # key needed\n",
                encoding="utf-8",
            )
            runner = FakeRunner()
            error_output = io.StringIO()
            with redirect_stderr(error_output):
                result = run_bootstrap(
                    ["--non-interactive"],
                    root=root,
                    runner=runner,
                    uv_path="uv",
                    environ={},
                    interactive=False,
                )
            self.assertEqual(result, 1)
            self.assertIn("OPENAI_API_KEY", error_output.getvalue())
            self.assertIn("--non-interactive", error_output.getvalue())
            self.assertFalse(any("main.py" in command for command in runner.commands))

    def test_interactive_default_selects_provider_and_launches_main(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = FakeRunner()
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                result = run_bootstrap(
                    [],
                    root=root,
                    runner=runner,
                    uv_path="uv",
                    input_fn=lambda prompt: "openai",
                    getpass_fn=lambda prompt: "openai-test-secret",
                    environ={},
                    interactive=True,
                )
            launch = runner.commands[-1]
            env_file = (root / ".env").read_text(encoding="utf-8")
            self.assertEqual(result, 0)
            self.assertIn("main.py", launch)
            self.assertEqual(launch[-2:], ["--provider", "openai"])
            self.assertNotIn("--mode", launch)
            self.assertNotIn("openai-test-secret", " ".join(launch))
            self.assertIn('OPENAI_API_KEY="openai-test-secret"', env_file)
            self.assertIn('LLM_PROVIDER="openai"', env_file)
            self.assertNotIn("openai-test-secret", output.getvalue())

    def test_explicit_mode_is_forwarded_to_main(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".env").write_text("OPENAI_API_KEY=existing\n", encoding="utf-8")
            runner = FakeRunner()
            result = run_bootstrap(
                ["--provider", "openai", "--mode", "stream"],
                root=root,
                runner=runner,
                uv_path="uv",
                environ={},
                interactive=False,
            )
            self.assertEqual(result, 0)
            self.assertEqual(runner.commands[-1][-2:], ["--mode", "stream"])

    def test_existing_python312_fallback_creates_venv_and_uses_pip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".env").write_text("OPENAI_API_KEY=existing\n", encoding="utf-8")
            runner = FakeRunner()
            result = run_bootstrap(
                ["--provider", "openai", "--setup-only"],
                root=root,
                runner=runner,
                uv_path="",
                which=lambda name: None,
                environ={},
                interactive=False,
            )
            self.assertEqual(result, 0)
            self.assertIn([sys.executable, "-m", "venv", ".venv"], runner.commands)
            self.assertTrue(any(command[1:4] == ["-m", "pip", "install"] for command in runner.commands))


if __name__ == "__main__":
    unittest.main()
