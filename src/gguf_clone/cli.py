from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from .common import OverwriteBehavior
from .main import run
from .params import extract_params


def _dispatch_run(args: argparse.Namespace) -> int:
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
    return run(
        config_path,
        verbose=cast(bool, args.verbose),
        overwrite_behavior=overwrite_behavior,
    )


def _dispatch_params(args: argparse.Namespace) -> int:
    directory = cast(Path, args.directory).expanduser().resolve()
    output = cast(Path | None, args.output)
    if output is not None:
        output = output.expanduser()
    return extract_params(
        directory,
        cast(list[str], args.patterns),
        output,
    )


def _is_run_fallback(argv: list[str]) -> bool:
    if not argv:
        return False
    if argv[0] in ("run", "params", "-h", "--help"):
        return False
    # Treat as implicit "run"
    return True


def main() -> None:
    parser = argparse.ArgumentParser(prog="gguf-clone")
    subparsers = parser.add_subparsers(dest="command")

    # -- run subcommand --
    run_parser = subparsers.add_parser(
        "run",
        help="run the full clone pipeline from a config file",
    )
    _ = run_parser.add_argument(
        "config",
        nargs="?",
        default="config.yml",
        type=Path,
        help="path to config YAML file (default: ./config.yml)",
    )
    _ = run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show verbose output",
    )
    overwrite_group = run_parser.add_mutually_exclusive_group()
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

    # -- params subcommand --
    params_parser = subparsers.add_parser(
        "params",
        help="extract quantization parameters from template GGUF(s)",
    )
    _ = params_parser.add_argument(
        "directory",
        type=Path,
        help="directory containing template GGUF file(s)",
    )
    _ = params_parser.add_argument(
        "patterns",
        nargs="*",
        default=["*.gguf"],
        help="glob pattern(s) to match GGUF files (default: *.gguf)",
    )
    _ = params_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="write JSON to file instead of stdout",
    )

    argv = sys.argv[1:]
    if _is_run_fallback(argv):
        argv = ["run", *argv]

    args = parser.parse_args(argv)
    command = cast(str | None, args.command)

    if command == "run":
        sys.exit(_dispatch_run(args))
    elif command == "params":
        sys.exit(_dispatch_params(args))
    else:
        parser.print_help()
        sys.exit(1)
