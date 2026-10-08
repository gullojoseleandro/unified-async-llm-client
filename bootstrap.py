from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
KEY_NAMES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
ENV_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


class BootstrapError(Exception):
    """An actionable local setup or configuration failure."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a Python 3.12 environment and run the async LLM example."
    )
    parser.add_argument("--provider", choices=("openai", "anthropic"), help="Select a provider instead of using saved configuration.")
    parser.add_argument("--mode", choices=("both", "normal", "stream"), help="Select a response mode; the default runs both.")
    parser.add_argument("--setup-only", action="store_true", help="Prepare dependencies and credentials without calling an API.")
    parser.add_argument("--non-interactive", action="store_true", help="Never prompt; fail with setup guidance when configuration is missing.")
    return parser


def _env_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ENV_ASSIGNMENT.match(line)
        if match:
            value = match.group(2).strip()
            if value.startswith(("\"", "'")):
                quote = value[0]
                closing_quote = value.find(quote, 1)
                value = value[1:closing_quote] if closing_quote > 0 else value[1:]
            else:
                value = "" if value.startswith("#") else re.split(r"\s+#", value, maxsplit=1)[0].strip()
            values[match.group(1)] = value
    return values


def _env_line_key(line: str) -> str | None:
    match = ENV_ASSIGNMENT.match(line)
    return match.group(1) if match else None


def _format_env_value(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _set_env_values(path: Path, updates: Mapping[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    present: set[str] = set()
    updated_lines: list[str] = []
    for line in lines:
        key = _env_line_key(line)
        if key in updates:
            updated_lines.append(f"{key}={_format_env_value(updates[key])}")
            present.add(key)
        else:
            updated_lines.append(line)
    updated_lines.extend(
        f"{key}={_format_env_value(value)}" for key, value in updates.items() if key not in present
    )
    path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)


def _venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def _checked_run(
    command: Sequence[str | Path],
    *,
    root: Path,
    runner: Callable[..., Any],
    purpose: str,
) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(
            [str(part) for part in command],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise BootstrapError(f"Could not {purpose}. Check that the required tool is installed and available on PATH.") from error
    if result.returncode != 0:
        raise BootstrapError(f"Could not {purpose}. Check your Python installation and network, then rerun the initializer.")
    return result


def _is_python312(result: subprocess.CompletedProcess[str]) -> bool:
    version = f"{result.stdout or ''} {result.stderr or ''}"
    return re.search(r"Python\s+3\.12(?:\.\d+)?\b", version) is not None


def _check_venv(venv_python: Path, *, root: Path, runner: Callable[..., Any]) -> None:
    if not venv_python.is_file():
        raise BootstrapError(
            f"The existing virtual environment is incomplete: {venv_python}. Remove that environment manually and rerun bootstrap.py."
        )
    result = _checked_run([venv_python, "--version"], root=root, runner=runner, purpose="check the virtual environment")
    if not _is_python312(result):
        raise BootstrapError(
            "The existing .venv is not Python 3.12. Remove that environment manually or rename it, then rerun bootstrap.py."
        )


def _find_python312(
    *,
    runner: Callable[..., Any],
    root: Path,
    which: Callable[[str], str | None],
) -> str:
    candidates: list[str] = []
    if sys.version_info[:2] == (3, 12):
        candidates.append(sys.executable)
    for name in ("python3.12", "python3", "python"):
        candidate = which(name)
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        result = _checked_run([candidate, "--version"], root=root, runner=runner, purpose="find Python 3.12")
        if _is_python312(result):
            return candidate
    raise BootstrapError(
        "Python 3.12 was not found and uv is unavailable. Install Python 3.12 or install uv, then rerun bootstrap.py."
    )


def _prepare_environment(
    *,
    root: Path,
    runner: Callable[..., Any],
    uv_path: str | None,
    which: Callable[[str], str | None],
) -> Path:
    venv_python = _venv_python(root)
    if (root / ".venv").exists():
        _check_venv(venv_python, root=root, runner=runner)
    elif uv_path:
        _checked_run([uv_path, "python", "install", "3.12"], root=root, runner=runner, purpose="install Python 3.12 with uv")
        _checked_run([uv_path, "venv", "--python", "3.12", ".venv"], root=root, runner=runner, purpose="create the Python 3.12 virtual environment")
        _check_venv(venv_python, root=root, runner=runner)
    else:
        python312 = _find_python312(runner=runner, root=root, which=which)
        _checked_run([python312, "-m", "venv", ".venv"], root=root, runner=runner, purpose="create the Python 3.12 virtual environment")
        _check_venv(venv_python, root=root, runner=runner)

    if uv_path:
        install_command = [uv_path, "pip", "install", "--python", str(venv_python), "-r", "requirements.txt"]
    else:
        install_command = [str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"]
    _checked_run(install_command, root=root, runner=runner, purpose="install project dependencies")
    return venv_python


def _select_provider(
    explicit: str | None,
    env_values: Mapping[str, str],
    environ: Mapping[str, str],
    *,
    interactive: bool,
    input_fn: Callable[[str], str],
) -> str:
    provider = explicit or environ.get("LLM_PROVIDER") or env_values.get("LLM_PROVIDER")
    if not provider and interactive:
        try:
            provider = input_fn("Choose a provider [openai/anthropic]: ").strip().lower()
        except (EOFError, KeyboardInterrupt) as error:
            raise BootstrapError("No provider was selected. Rerun bootstrap.py and choose openai or anthropic.") from error
    if provider not in KEY_NAMES:
        raise BootstrapError("Choose a provider with --provider openai|anthropic or set LLM_PROVIDER in .env.")
    return provider


def _get_api_key(
    provider: str,
    env_values: Mapping[str, str],
    environ: Mapping[str, str],
    *,
    interactive: bool,
    getpass_fn: Callable[[str], str],
) -> tuple[str | None, bool]:
    key_name = KEY_NAMES[provider]
    existing = environ.get(key_name) or env_values.get(key_name)
    if existing:
        return existing, False
    if not interactive:
        raise BootstrapError(
            f"No {key_name} is configured. Set it in .env or the environment, or rerun without --non-interactive to enter it securely."
        )
    try:
        value = getpass_fn(f"Enter {key_name} (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt) as error:
        raise BootstrapError(f"No {key_name} was entered. Rerun bootstrap.py to configure this provider.") from error
    if not value:
        raise BootstrapError(f"No {key_name} was entered. Rerun bootstrap.py to configure this provider.")
    return value, True


def run_bootstrap(
    argv: Sequence[str] | None = None,
    *,
    root: Path | None = None,
    runner: Callable[..., Any] | None = None,
    input_fn: Callable[[str], str] | None = None,
    getpass_fn: Callable[[str], str] | None = None,
    environ: Mapping[str, str] | None = None,
    interactive: bool | None = None,
    uv_path: str | None = None,
    which: Callable[[str], str | None] | None = None,
    output: Any = None,
    error_output: Any = None,
) -> int:
    args = _parser().parse_args(argv)
    root = (root or ROOT).resolve()
    runner = runner or subprocess.run
    input_fn = input_fn or input
    getpass_fn = getpass_fn or getpass.getpass
    environ = environ if environ is not None else os.environ
    which = which or shutil.which
    output = output or sys.stdout
    error_output = error_output or sys.stderr
    interactive = interactive if interactive is not None else (
        not args.non_interactive and sys.stdin.isatty() and sys.stdout.isatty()
    )
    uv_path = uv_path if uv_path is not None else which("uv")
    env_path = root / ".env"

    try:
        env_values = _env_values(env_path)
        provider = _select_provider(
            args.provider,
            env_values,
            environ,
            interactive=interactive,
            input_fn=input_fn,
        )
        key_name = KEY_NAMES[provider]
        key_exists = bool(environ.get(key_name) or env_values.get(key_name))
        if not key_exists and not interactive:
            raise BootstrapError(
                f"No {key_name} is configured. Set it in .env or the environment, or rerun without --non-interactive to enter it securely."
            )

        venv_python = _prepare_environment(root=root, runner=runner, uv_path=uv_path, which=which)
        api_key, prompted = _get_api_key(
            provider,
            env_values,
            environ,
            interactive=interactive,
            getpass_fn=getpass_fn,
        )
        updates = {"LLM_PROVIDER": provider}
        if prompted and api_key is not None:
            updates[key_name] = api_key
        _set_env_values(env_path, updates)
        print(f"Prepared Python 3.12 environment for {provider}.", file=output)

        if args.setup_only:
            print("Setup complete. Run bootstrap.py again to start the example.", file=output)
            return 0

        command = [str(venv_python), "main.py", "--provider", provider]
        if args.mode:
            command.extend(("--mode", args.mode))
        print(f"Starting the {provider} example.", file=output)
        try:
            result = runner(command, cwd=str(root), check=False)
        except OSError as error:
            raise BootstrapError("Could not launch main.py from the Python 3.12 environment.") from error
        return int(result.returncode)
    except BootstrapError as error:
        print(f"Setup error: {error}", file=error_output)
        return 1
    except OSError:
        print("Setup error: Could not read or update .env. Check local file permissions.", file=error_output)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    return run_bootstrap(argv)


if __name__ == "__main__":
    raise SystemExit(main())
