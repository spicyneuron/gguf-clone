from __future__ import annotations

from pathlib import Path

from .common import log_line, run_command


def build_quantize_options(tensor_types: list[str], imatrix: str) -> list[str]:
    options: list[str] = []
    if imatrix:
        options.extend(["--imatrix", imatrix])
    for tensor_type in tensor_types:
        options.extend(["--tensor-type", tensor_type])
    return options


def quantize_gguf(
    input_path: Path,
    output_path: Path,
    *,
    tensor_types: list[str],
    default_type: str,
    imatrix: str,
    llama_quantize: Path,
    cwd: Path | None = None,
    indent: str = "",
) -> int:
    options = build_quantize_options(tensor_types, imatrix)

    cmd = [
        str(llama_quantize),
        *options,
        str(input_path),
        str(output_path),
        default_type,
    ]

    log_line(f"Running: {' '.join(cmd)}", indent=indent)
    return run_command(cmd, cwd=cwd)
