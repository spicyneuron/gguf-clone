from __future__ import annotations

import sys
from pathlib import Path

from .common import confirm_overwrite, log_line, log_success, remove_files, run_command
from .gguf_path import build_gguf_env


def convert_target(
    model_path: Path,
    *,
    output_dir: Path,
    outfile_name: str | None,
    existing_glob: str,
    convert_script: Path,
    label: str = "converted",
    cwd: Path | None = None,
    indent: str = "",
    use_message: str | None = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = list(output_dir.glob(existing_glob))
    action = "overwrite"
    if existing_outputs:
        action = confirm_overwrite(existing_outputs, label, indent=indent)
        if action == "cancel":
            return 1
        if action == "use":
            if use_message:
                log_success(use_message, indent=indent)
            return 0
        if not remove_files(existing_outputs):
            return 1

    outfile_arg = str(output_dir / outfile_name) if outfile_name else str(output_dir)
    cmd = [
        sys.executable,
        str(convert_script),
        "--outtype",
        "auto",
        "--outfile",
        outfile_arg,
        str(model_path),
    ]

    log_line(f"Running: {' '.join(cmd)}", indent=indent)
    returncode = run_command(cmd, cwd=cwd, env=build_gguf_env())
    if returncode != 0:
        return returncode

    log_success(f"Converted output saved to {outfile_arg}", indent=indent)
    return 0
