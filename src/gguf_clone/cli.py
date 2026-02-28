from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from . import config as config_mod
from .artifacts import Artifacts
from .common import OverwriteBehavior
from .main import run_extract_template, run_pipeline, run_quantize_gguf, run_quantize_mlx


def _overwrite_from_args(args: argparse.Namespace) -> OverwriteBehavior | None:
    if cast(bool, args.use_existing):
        return "use"
    if cast(bool, args.overwrite):
        return "overwrite"
    if cast(bool, args.cancel):
        return "cancel"
    return None


def _load_config_and_artifacts(
    config_path: Path,
) -> tuple[config_mod.RunConfig, Artifacts] | None:
    config = config_mod.load_config(config_path)
    if not config:
        return None
    artifacts = Artifacts.from_config(config.output_dir, config.template, config.target)
    return config, artifacts


def _dispatch_run(args: argparse.Namespace) -> int:
    config_path = cast(Path, args.config).expanduser()
    return run_pipeline(
        config_path,
        verbose=cast(bool, args.verbose),
        overwrite_behavior=_overwrite_from_args(args),
    )


def _dispatch_extract_template(args: argparse.Namespace) -> int:
    config_path = cast(Path, args.config).expanduser()
    loaded = _load_config_and_artifacts(config_path)
    if not loaded:
        return 1
    config, artifacts = loaded
    if config.extract_template is None:
        print("extract_template section missing from config.")
        return 1
    return run_extract_template(
        config,
        artifacts,
        verbose=cast(bool, args.verbose),
        overwrite_behavior=_overwrite_from_args(args),
    )


def _dispatch_quantize_gguf(args: argparse.Namespace) -> int:
    config_path = cast(Path, args.config).expanduser()
    loaded = _load_config_and_artifacts(config_path)
    if not loaded:
        return 1
    config, artifacts = loaded
    if config.quantize_gguf is None:
        print("quantize_gguf section missing from config.")
        return 1
    return run_quantize_gguf(
        config,
        artifacts,
        verbose=cast(bool, args.verbose),
        overwrite_behavior=_overwrite_from_args(args),
    )


def _dispatch_quantize_mlx(args: argparse.Namespace) -> int:
    config_path = cast(Path, args.config).expanduser()
    loaded = _load_config_and_artifacts(config_path)
    if not loaded:
        return 1
    config, artifacts = loaded
    if config.quantize_mlx is None:
        print("quantize_mlx section missing from config.")
        return 1
    return run_quantize_mlx(
        config,
        artifacts,
        verbose=cast(bool, args.verbose),
        overwrite_behavior=_overwrite_from_args(args),
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "config",
        nargs="?",
        default="config.yml",
        type=Path,
        help="path to config YAML file (default: ./config.yml)",
    )
    _ = parser.add_argument(
        "-v", "--verbose", action="store_true", help="show verbose output"
    )
    group = parser.add_mutually_exclusive_group()
    _ = group.add_argument(
        "--use-existing",
        action="store_true",
        help="use existing outputs without prompting",
    )
    _ = group.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing outputs without prompting",
    )
    _ = group.add_argument(
        "--cancel",
        action="store_true",
        help="cancel immediately if outputs already exist",
    )


SUBCOMMANDS = ("run", "extract-template", "quantize-gguf", "quantize-mlx")


def _is_run_fallback(argv: list[str]) -> bool:
    if not argv:
        return False
    if argv[0] in (*SUBCOMMANDS, "-h", "--help"):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(prog="gguf-clone")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run", help="run declared pipeline stages in order"
    )
    _add_common_args(run_parser)

    ep_parser = subparsers.add_parser(
        "extract-template", help="extract quantization parameters from template GGUFs"
    )
    _add_common_args(ep_parser)

    qg_parser = subparsers.add_parser(
        "quantize-gguf", help="quantize target model to GGUF using extracted params"
    )
    _add_common_args(qg_parser)

    qm_parser = subparsers.add_parser(
        "quantize-mlx", help="quantize target model to MLX using extracted params"
    )
    _add_common_args(qm_parser)

    argv = sys.argv[1:]
    if _is_run_fallback(argv):
        argv = ["run", *argv]

    args = parser.parse_args(argv)
    command = cast(str | None, args.command)

    dispatch = {
        "run": _dispatch_run,
        "extract-template": _dispatch_extract_template,
        "quantize-gguf": _dispatch_quantize_gguf,
        "quantize-mlx": _dispatch_quantize_mlx,
    }

    handler = dispatch.get(command or "")
    if handler:
        sys.exit(handler(args))
    else:
        parser.print_help()
        sys.exit(1)
