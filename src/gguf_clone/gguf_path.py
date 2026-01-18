from __future__ import annotations

import os
import sys
from pathlib import Path

LLAMA_CPP_DIR_ENV = "GGUF_CLONE_LLAMA_CPP"


def configure_gguf_path() -> None:
    root = resolve_gguf_root()
    if root:
        _prepend_sys_path(root)


def resolve_gguf_root() -> Path | None:
    llama_cpp_dir = os.getenv(LLAMA_CPP_DIR_ENV)
    if llama_cpp_dir:
        root = _normalize_gguf_root(Path(llama_cpp_dir).expanduser() / "gguf-py")
        if root:
            return root

    vendored_root = _resolve_vendored_gguf_root()
    if vendored_root:
        return vendored_root

    package_root = _normalize_gguf_root(Path(__file__).resolve().parents[1])
    if package_root:
        return package_root

    return None


def build_gguf_env() -> dict[str, str] | None:
    root = resolve_gguf_root()
    if not root:
        return None
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    prefix = str(root)
    if existing:
        paths = existing.split(os.pathsep)
        if prefix not in paths:
            env["PYTHONPATH"] = os.pathsep.join([prefix, *paths])
    else:
        env["PYTHONPATH"] = prefix
    return env


def _normalize_gguf_root(path: Path) -> Path | None:
    package_root = path / "gguf"
    if (package_root / "__init__.py").exists():
        return path
    if path.name == "gguf" and (path / "__init__.py").exists():
        return path.parent
    return None


def _resolve_vendored_gguf_root() -> Path | None:
    vendored_root = _normalize_gguf_root(Path(__file__).resolve().parent / "_vendor")
    if vendored_root:
        return vendored_root
    return None


def _prepend_sys_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
