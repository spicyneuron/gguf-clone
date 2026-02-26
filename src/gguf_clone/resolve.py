from __future__ import annotations

import fnmatch
import importlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import SourceRef

if TYPE_CHECKING:

    def _hf_hub_download(*_args: object, **_kwargs: object) -> str: ...
    def _snapshot_download(*_args: object, **_kwargs: object) -> str: ...
else:
    from huggingface_hub import hf_hub_download as _hf_hub_download
    from huggingface_hub import snapshot_download as _snapshot_download


DEPENDENCIES = [
    "numpy",
    "sentencepiece",
    "transformers",
    "google.protobuf",
    "gguf",
    "huggingface_hub",
    "yaml",
    "torch",
]


LLAMA_CPP_DIR_ENV = "GGUF_CLONE_LLAMA_CPP"


class ToolResolutionError(RuntimeError):
    pass


hf_hub_download = _hf_hub_download
snapshot_download = _snapshot_download


@dataclass(frozen=True)
class ToolPaths:
    llama_quantize: Path
    llama_gguf_split: Path
    convert_hf_to_gguf: Path


def _env_path(name: str) -> Path | None:
    value = os.getenv(name)
    if not value:
        return None
    return Path(value).expanduser()


def _resolve_from_path(command: str) -> Path | None:
    resolved = shutil.which(command)
    if not resolved:
        return None
    return Path(resolved)


def _resolve_from_llama_cpp_dir(command: str, *, require_exec: bool) -> Path | None:
    root = _env_path(LLAMA_CPP_DIR_ENV)
    if not root:
        return None

    candidates = [
        root / command,
        root / "bin" / command,
        root / "build" / "bin" / command,
    ]

    for candidate in candidates:
        if candidate.exists() and (not require_exec or os.access(candidate, os.X_OK)):
            return candidate

    return None


def _resolve_vendored_convert() -> Path | None:
    candidate = (
        Path(__file__).resolve().parent
        / "_vendor"
        / "llama_cpp"
        / "convert_hf_to_gguf.py"
    )
    if candidate.exists():
        return candidate
    return None


def resolve_llama_quantize() -> Path:
    path = _resolve_from_llama_cpp_dir("llama-quantize", require_exec=True)
    if path:
        return path

    path = _resolve_from_path("llama-quantize")
    if path:
        return path

    raise ToolResolutionError(
        "llama-quantize not found. Set GGUF_CLONE_LLAMA_CPP or add it to PATH."
    )


def resolve_llama_gguf_split() -> Path:
    path = _resolve_from_llama_cpp_dir("llama-gguf-split", require_exec=True)
    if path:
        return path

    path = _resolve_from_path("llama-gguf-split")
    if path:
        return path

    raise ToolResolutionError(
        "llama-gguf-split not found. Set GGUF_CLONE_LLAMA_CPP or add it to PATH."
    )


def resolve_convert_hf_to_gguf() -> Path:
    path = _resolve_from_llama_cpp_dir("convert_hf_to_gguf.py", require_exec=False)
    if path:
        return path

    path = _resolve_vendored_convert()
    if path:
        return path

    raise ToolResolutionError(
        "convert_hf_to_gguf.py not found. Set GGUF_CLONE_LLAMA_CPP "
        + "or use the vendored copy."
    )


def resolve_tools() -> ToolPaths:
    return ToolPaths(
        llama_quantize=resolve_llama_quantize(),
        llama_gguf_split=resolve_llama_gguf_split(),
        convert_hf_to_gguf=resolve_convert_hf_to_gguf(),
    )


def check_deps() -> list[str]:
    missing: list[str] = []
    for name in DEPENDENCIES:
        try:
            _ = importlib.import_module(name)
        except Exception:
            missing.append(name)
    return missing


def check_gguf_support() -> str | None:
    try:
        vocab = importlib.import_module("gguf.vocab")
    except Exception:
        return "Failed to import gguf.vocab."

    if not hasattr(vocab, "MistralTokenizerType"):
        return (
            "gguf package is missing MistralTokenizerType. Use the vendored gguf or set "
            + "GGUF_CLONE_LLAMA_CPP to a newer gguf-py."
        )

    return None


def print_env_hint() -> None:
    envs = [
        "GGUF_CLONE_LLAMA_CPP",
    ]
    print("Environment variables:")
    for env in envs:
        value = os.getenv(env)
        suffix = f"={value}" if value else "(not set)"
        print(f"  {env} {suffix}")


class ModelResolutionError(RuntimeError):
    pass


def _resolve_local_source(path: Path, label: str) -> Path:
    resolved = path.expanduser()
    if not resolved.exists():
        raise ModelResolutionError(f"{label} path not found: {resolved}")
    return resolved


def resolve_hf_repo(
    repo_id: str,
    *,
    revision: str | None = None,
    filename: str | None = None,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    local_files_only: bool = False,
) -> Path:
    try:
        if filename:
            resolved = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_files_only=local_files_only,
            )
            return Path(resolved)

        resolved = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=local_files_only,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        return Path(resolved)
    except Exception as exc:
        raise ModelResolutionError(f"Failed to resolve '{repo_id}': {exc}") from exc


def filter_paths(
    base_dir: Path,
    candidates: list[Path],
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
) -> list[Path]:
    if allow_patterns:
        allowed: list[Path] = []
        for path in candidates:
            rel = path.relative_to(base_dir).as_posix()
            if any(fnmatch.fnmatch(rel, pattern) for pattern in allow_patterns):
                allowed.append(path)
        candidates = allowed

    if ignore_patterns:
        filtered: list[Path] = []
        for path in candidates:
            rel = path.relative_to(base_dir).as_posix()
            if not any(fnmatch.fnmatch(rel, pattern) for pattern in ignore_patterns):
                filtered.append(path)
        candidates = filtered

    return sorted(candidates)


def match_pattern(
    base_dir: Path,
    candidates: list[Path],
    pattern: str,
    label: str,
    *,
    require_single: bool = False,
) -> list[Path] | None:
    matches = filter_paths(base_dir, candidates, [pattern], None)
    if not matches:
        print(f"No matches found for {label} pattern: {pattern}")
        return None
    if require_single and len(matches) > 1:
        print(f"{label} pattern matched {len(matches)} files:")
        for path in matches[:5]:
            print(f"  {path}")
        if len(matches) > 5:
            print(f"  ...and {len(matches) - 5} more")
        return None
    return matches


def resolve_source_snapshot(
    ref: SourceRef,
    *,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> Path:
    """Resolve a SourceRef to a local directory path.

    Local paths are validated; HF repos are downloaded (with caching).
    """
    if ref.path is not None:
        return _resolve_local_source(ref.path, "Source")
    if ref.repo is None:
        raise ModelResolutionError("Source has no repo or path configured.")
    return resolve_hf_repo(
        ref.repo,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
