from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from . import config as config_mod
from .artifacts import Artifacts
from .common import OverwriteBehavior, set_overwrite_behavior, set_verbose
from .resolve import check_deps, check_gguf_support
from .stages import run_extract_template_stage, run_quantize_gguf_stage


def _init_run(
    *,
    verbose: bool,
    overwrite_behavior: OverwriteBehavior | None,
) -> str | None:
    set_verbose(verbose)
    set_overwrite_behavior(overwrite_behavior)

    missing = check_deps()
    if missing:
        return "Missing Python dependencies: " + ", ".join(missing)

    gguf_error = check_gguf_support()
    if gguf_error:
        return gguf_error

    return None


def run_extract_template(
    config: config_mod.RunConfig,
    artifacts: Artifacts,
    *,
    verbose: bool = False,
    overwrite_behavior: OverwriteBehavior | None = None,
) -> int:
    err = _init_run(verbose=verbose, overwrite_behavior=overwrite_behavior)
    if err:
        print(err)
        return 1
    return run_extract_template_stage(config, artifacts)


def run_quantize_gguf(
    config: config_mod.RunConfig,
    artifacts: Artifacts,
    *,
    verbose: bool = False,
    overwrite_behavior: OverwriteBehavior | None = None,
) -> int:
    err = _init_run(verbose=verbose, overwrite_behavior=overwrite_behavior)
    if err:
        print(err)
        return 1
    return run_quantize_gguf_stage(config, artifacts)


def run_quantize_mlx(
    config: config_mod.RunConfig,
    artifacts: Artifacts,
    *,
    verbose: bool = False,
    overwrite_behavior: OverwriteBehavior | None = None,
) -> int:
    del artifacts
    err = _init_run(verbose=verbose, overwrite_behavior=overwrite_behavior)
    if err:
        print(err)
        return 1
    if config.quantize_mlx is None:
        print("quantize_mlx section missing from config.")
        return 1
    print("quantize_mlx stage is declared but not implemented yet.")
    return 1


def run_pipeline(
    config_path: Path,
    *,
    verbose: bool = False,
    overwrite_behavior: OverwriteBehavior | None = None,
) -> int:
    config_path = config_path.expanduser()
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return 1

    config = config_mod.load_config(config_path)
    if not config:
        return 1

    artifacts = Artifacts.from_config(config.output_dir, config.template, config.target)

    stages: list[tuple[object, Callable[..., int]]] = [
        (config.extract_template, run_extract_template),
        (config.quantize_gguf, run_quantize_gguf),
        (config.quantize_mlx, run_quantize_mlx),
    ]

    ran_any = False
    for section, fn in stages:
        if section is None:
            continue
        ran_any = True
        result: int = fn(
            config,
            artifacts,
            verbose=verbose,
            overwrite_behavior=overwrite_behavior,
        )
        if result != 0:
            return result

    if not ran_any:
        print("No stages declared in config (extract_template, quantize_gguf, quantize_mlx).")
        return 1

    return 0
