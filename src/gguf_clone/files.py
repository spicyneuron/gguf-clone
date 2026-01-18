from __future__ import annotations

import shutil
from pathlib import Path

from .common import confirm_overwrite, log_line, log_success
from .resolve import filter_paths


def _match_copy_files(
    template_snapshot: Path, patterns: list[str]
) -> list[Path] | None:
    if not patterns:
        return []

    candidates = [path for path in template_snapshot.rglob("*") if path.is_file()]
    matches: list[Path] = []
    for pattern in patterns:
        pattern_matches = filter_paths(template_snapshot, candidates, [pattern], None)
        if not pattern_matches:
            print(f"No matches found for template copy_files pattern: {pattern}")
            return None
        matches.extend(pattern_matches)

    return list(dict.fromkeys(matches))


def copy_template_files(
    template_snapshot: Path,
    patterns: list[str],
    output_dir: Path,
    *,
    indent: str = "",
) -> int:
    matches = _match_copy_files(template_snapshot, patterns)
    if matches is None:
        return 1
    if not matches:
        return 0

    outputs: dict[Path, Path] = {}
    for source in matches:
        dest = output_dir / source.name
        if dest in outputs:
            print(
                "Template copy_files patterns matched multiple files with the same name:"
            )
            print(f"  {outputs[dest]}")
            print(f"  {source}")
            return 1
        outputs[dest] = source

    log_line(
        f"Copying {len(outputs)} template file(s) to {output_dir}",
        indent=indent,
    )

    action = confirm_overwrite(list(outputs.keys()), "copied file", indent=indent)
    if action == "cancel":
        return 1

    for dest, source in outputs.items():
        if dest.exists() and action == "use":
            continue
        _ = shutil.copy2(source, dest)

    log_success(
        f"Files copied to {output_dir}",
        indent=indent,
    )
    return 0
