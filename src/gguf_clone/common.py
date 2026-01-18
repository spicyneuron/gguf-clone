from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Literal, cast

_verbose = False
OverwriteBehavior = Literal["use", "overwrite", "cancel"]
_overwrite_behavior: OverwriteBehavior | None = None


def set_verbose(value: bool) -> None:
    global _verbose
    _verbose = value


def set_overwrite_behavior(value: OverwriteBehavior | None) -> None:
    global _overwrite_behavior
    _overwrite_behavior = value


def is_verbose() -> bool:
    return _verbose


def print_existing_files(paths: list[Path], label: str, indent: str = "") -> None:
    if not paths:
        return
    if len(paths) == 1:
        header = f"Existing {label} file found:"
    else:
        header = f"Existing {label} files found ({len(paths)}):"
    print(f"{indent}{header}")
    list_indent = indent or "  "
    for path in paths[:5]:
        print(f"{list_indent}{path}")
    if len(paths) > 5:
        print(f"{list_indent}...and {len(paths) - 5} more")


def log_stage(title: str) -> str:
    if not _verbose:
        return ""
    print(f"{title}:")
    return "  "


def log_line(message: str, *, indent: str = "  ") -> None:
    if not _verbose:
        return
    print(f"{indent}{message}")


def log_success(message: str, *, indent: str = "") -> None:
    print(f"{indent}{message}")


def confirm_overwrite(paths: list[Path], label: str, *, indent: str = "") -> str:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return "overwrite"
    print_existing_files(existing, label, indent)
    if _overwrite_behavior == "use":
        return "use"
    if _overwrite_behavior == "overwrite":
        return "overwrite"
    if _overwrite_behavior == "cancel":
        print(f"{indent}Cancel enabled; refusing to overwrite existing {label} files.")
        return "cancel"
    if not sys.stdin.isatty():
        print(
            f"{indent}Non-interactive mode requires --use-existing, --overwrite, or --cancel."
        )
        return "cancel"
    response = input("[u]se existing (default)/[o]verwrite/[c]ancel? ").strip().lower()
    if response in ("", "u", "use", "use existing"):
        return "use"
    if response in ("o", "overwrite"):
        return "overwrite"
    if response in ("c", "cancel"):
        print(f"{indent}Cancelled by user.")
        return "cancel"
    print(f"{indent}Cancelled by user.")
    return "cancel"


def remove_files(paths: list[Path]) -> bool:
    for path in paths:
        if not path.exists():
            continue
        if not path.is_file():
            print(f"Refusing to remove non-file: {path}")
            return False
        path.unlink()
    return True


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run a command, capturing output unless verbose mode is enabled.

    Returns the process return code. Prints captured output on failure.
    """
    if _verbose:
        result = subprocess.run(cmd, cwd=cwd, env=env)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)

    if result.returncode != 0 and not _verbose:
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip())

    return result.returncode or 0


def require_mapping(value: object, message: str) -> dict[str, object] | None:
    if not isinstance(value, dict):
        print(message)
        return None
    raw = cast(dict[object, object], value)
    mapping: dict[str, object] = {}
    for key, item in raw.items():
        if not isinstance(key, str):
            print(message)
            return None
        mapping[key] = item
    return mapping
