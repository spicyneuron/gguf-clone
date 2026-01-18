from __future__ import annotations

import re
from pathlib import Path

from .common import log_line, run_command

_SIZE_UNITS = {
    "M": 1024 * 1024,
    "G": 1024 * 1024 * 1024,
}


def parse_split_size(value: str) -> int | None:
    if not value or not value.strip():
        return None
    match = re.fullmatch(r"(?P<size>\d+)(?P<unit>[mMgG])?", value.strip())
    if not match:
        return None
    size = int(match.group("size"))
    unit = match.group("unit")
    if unit:
        size *= _SIZE_UNITS[unit.upper()]
    return size


def split_needed(file_size: int, max_size: str) -> bool | None:
    limit = parse_split_size(max_size)
    if limit is None:
        return None
    return file_size > limit


def build_split_options(max_size: str) -> list[str]:
    return ["--split", "--split-max-size", max_size]


def split_gguf(
    input_path: Path,
    output_path: Path,
    *,
    max_size: str,
    llama_gguf_split: Path,
    cwd: Path | None = None,
    indent: str = "",
) -> int:
    options = build_split_options(max_size)

    cmd = [
        str(llama_gguf_split),
        *options,
        str(input_path),
        str(output_path),
    ]

    log_line(f"Running: {' '.join(cmd)}", indent=indent)
    return run_command(cmd, cwd=cwd)
