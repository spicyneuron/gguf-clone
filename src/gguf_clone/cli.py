from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from .common import OverwriteBehavior
from .main import run


def main() -> None:
    parser = argparse.ArgumentParser(prog="gguf-clone")
    _ = parser.add_argument(
        "config",
        nargs="?",
        default="config.yml",
        type=Path,
        help="path to config YAML file (default: ./config.yml)",
    )
    _ = parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show verbose output",
    )
    overwrite_group = parser.add_mutually_exclusive_group()
    _ = overwrite_group.add_argument(
        "--use-existing",
        action="store_true",
        help="use existing outputs without prompting",
    )
    _ = overwrite_group.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing outputs without prompting",
    )
    _ = overwrite_group.add_argument(
        "--cancel",
        action="store_true",
        help="cancel immediately if outputs already exist",
    )
    args = parser.parse_args()
    config_path = cast(Path, args.config).expanduser()
    use_existing = cast(bool, args.use_existing)
    overwrite = cast(bool, args.overwrite)
    cancel = cast(bool, args.cancel)
    overwrite_behavior: OverwriteBehavior | None = None
    if use_existing:
        overwrite_behavior = "use"
    elif overwrite:
        overwrite_behavior = "overwrite"
    elif cancel:
        overwrite_behavior = "cancel"
    sys.exit(
        run(
            config_path,
            verbose=cast(bool, args.verbose),
            overwrite_behavior=overwrite_behavior,
        )
    )
