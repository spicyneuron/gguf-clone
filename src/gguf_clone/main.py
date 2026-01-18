from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import yaml
from pydantic import BaseModel, BeforeValidator, ValidationError

from .common import (
    OverwriteBehavior,
    confirm_overwrite,
    log_line,
    log_stage,
    log_success,
    remove_files,
    set_overwrite_behavior,
    set_verbose,
)
from .convert import convert_target
from .files import copy_template_files
from .metadata import apply_metadata, copy_template_metadata
from .params import build_params_payload, copy_imatrix, load_params, save_params_payload
from .quantize import quantize_gguf
from .resolve import (
    ToolResolutionError,
    check_deps,
    check_gguf_support,
    print_env_hint,
    resolve_models,
    resolve_tools,
)
from .split import split_gguf, split_needed


@dataclass(frozen=True)
class RunConfig:
    template_repo: str
    template_gguf_patterns: list[str]
    template_imatrix_pattern: str
    template_copy_metadata: list[str]
    template_copy_files: list[str]
    target_repo: str
    output_prefix: str
    output_split: str
    output_converted_dir: str
    output_params_dir: str
    output_quantized_dir: str
    output_apply_metadata: dict[str, str]


def _ensure_list(v: str | list[str]) -> list[str]:
    return [v] if isinstance(v, str) else v


StrList = Annotated[list[str], BeforeValidator(_ensure_list)]


class TemplateConfig(BaseModel):
    repo: str
    ggufs: StrList
    imatrix: str
    copy_metadata: StrList = []
    copy_files: StrList = []


class TargetConfig(BaseModel):
    repo: str


class OutputConfig(BaseModel):
    prefix: str = ""
    split: str = "50G"
    converted_dir: str = "converted"
    params_dir: str = "params"
    quantized_dir: str = "quantized"
    apply_metadata: dict[str, str] = {
        "general.quantized_by": "https://github.com/spicyneuron/gguf-clone"
    }


class ConfigFile(BaseModel):
    template: TemplateConfig
    target: TargetConfig
    output: OutputConfig = OutputConfig()


def load_config(path: Path) -> RunConfig | None:
    try:
        data = cast(object, yaml.safe_load(path.read_text()))
        config = ConfigFile.model_validate(data)
    except (yaml.YAMLError, ValidationError) as exc:
        print(f"Failed to read config: {exc}")
        return None

    return RunConfig(
        template_repo=config.template.repo,
        template_gguf_patterns=config.template.ggufs,
        template_imatrix_pattern=config.template.imatrix,
        template_copy_metadata=config.template.copy_metadata,
        template_copy_files=config.template.copy_files,
        target_repo=config.target.repo,
        output_prefix=config.output.prefix,
        output_split=config.output.split,
        output_converted_dir=config.output.converted_dir,
        output_params_dir=config.output.params_dir,
        output_quantized_dir=config.output.quantized_dir,
        output_apply_metadata=config.output.apply_metadata,
    )


def repo_slug(repo_id: str) -> str:
    return repo_id.replace("/", "-")


def prefix_slug(prefix: str) -> str:
    return "-".join(prefix.strip().split())


def quant_label_from_stem(stem: str) -> str | None:
    ud = re.search(r"(UD-[A-Z0-9_]+)", stem, re.IGNORECASE)
    if ud:
        return ud.group(1).upper()

    patterns = [
        re.compile(r"(IQ\d+_[A-Z0-9_]+)", re.IGNORECASE),
        re.compile(r"(Q\d+_[A-Z0-9_]+)", re.IGNORECASE),
        re.compile(r"(BF16|F16|F32)", re.IGNORECASE),
    ]
    for pattern in patterns:
        matches: list[str] = pattern.findall(stem)
        if matches:
            return matches[-1].upper()
    return None


def _collect_split_outputs(output_path: Path, quant_label: str) -> list[Path]:
    split_dir = output_path.parent / quant_label
    if not split_dir.exists():
        return []
    return sorted(split_dir.glob(f"{output_path.stem}-*-of-*.gguf"))


def run(
    config_path: Path,
    *,
    verbose: bool = False,
    overwrite_behavior: OverwriteBehavior | None = None,
) -> int:
    set_verbose(verbose)
    set_overwrite_behavior(overwrite_behavior)
    missing = check_deps()
    if missing:
        print("Missing Python dependencies:")
        for name in missing:
            print(f"  {name}")
        print("\nInstall dependencies with: uv sync")
        return 1

    gguf_error = check_gguf_support()
    if gguf_error:
        print(gguf_error)
        print("\nInstall dependencies with: uv sync")
        return 1

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return 1

    config = load_config(config_path)
    if not config:
        return 1

    try:
        tools = resolve_tools()
    except ToolResolutionError as exc:
        print(str(exc))
        print_env_hint()
        return 1

    work_dir = config_path.parent
    converted_dir = work_dir / config.output_converted_dir
    params_dir = work_dir / config.output_params_dir
    quantized_dir = work_dir / config.output_quantized_dir

    converted_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)
    quantized_dir.mkdir(parents=True, exist_ok=True)

    stage_indent = log_stage("Validating inputs")
    log_line(f"Template repo: {config.template_repo}", indent=stage_indent)
    log_line(
        f"Template GGUFs: {', '.join(config.template_gguf_patterns)}",
        indent=stage_indent,
    )
    log_line(
        f"Template imatrix: {config.template_imatrix_pattern}", indent=stage_indent
    )
    if config.template_copy_metadata:
        log_line(
            f"Template metadata copy: {', '.join(config.template_copy_metadata)}",
            indent=stage_indent,
        )
    if config.template_copy_files:
        log_line(
            f"Template file copy: {', '.join(config.template_copy_files)}",
            indent=stage_indent,
        )
    log_line(f"Target repo: {config.target_repo}", indent=stage_indent)
    log_line(f"Output prefix: {config.output_prefix}", indent=stage_indent)
    log_line(f"Output split: {config.output_split}", indent=stage_indent)

    resolved = resolve_models(
        template_repo=config.template_repo,
        template_gguf_patterns=config.template_gguf_patterns,
        template_imatrix_pattern=config.template_imatrix_pattern,
        template_copy_files=config.template_copy_files,
        target_repo=config.target_repo,
    )
    if not resolved:
        return 1

    prefix = prefix_slug(config.output_prefix)
    target_repo_slug = repo_slug(config.target_repo)
    converted_name = f"{target_repo_slug}.gguf"

    stage_indent = log_stage("Converting target model to GGUF")
    result = convert_target(
        resolved.target_snapshot,
        output_dir=converted_dir,
        outfile_name=converted_name,
        existing_glob=converted_name,
        convert_script=tools.convert_hf_to_gguf,
        cwd=work_dir,
        indent=stage_indent,
        use_message="Using existing converted outputs.",
    )
    if result != 0:
        return result

    converted_gguf = converted_dir / converted_name
    if not converted_gguf.exists():
        print(f"Converted GGUF not found: {converted_gguf}")
        return 1

    # Copy imatrix to workspace so GGUF metadata uses a relative path
    imatrix_rel_path = copy_imatrix(
        resolved.template_imatrix, params_dir, prefix=config.output_params_dir
    )

    template_repo_slug = repo_slug(config.template_repo)
    for template_group in resolved.template_ggufs:
        shard_suffix = ""
        if len(template_group) > 1:
            shard_suffix = f" (+{len(template_group) - 1} shards)"
        stage_indent = log_stage(
            f"Building params from template GGUFs: {template_group[0].name}{shard_suffix}"
        )
        stem_labels = {
            label
            for path in template_group
            if (label := quant_label_from_stem(path.stem))
        }
        if len(stem_labels) > 1:
            print("Template GGUF pattern matched multiple quant labels:")
            for label in sorted(stem_labels):
                print(f"  {label}")
            return 1
        stem_label = next(iter(stem_labels), None)
        payload, params = build_params_payload(template_group, imatrix_rel_path)
        quant_label = stem_label or params.default_type
        params_path = params_dir / f"{template_repo_slug}-{quant_label}.json"
        action = confirm_overwrite([params_path], "params", indent=stage_indent)
        if action == "cancel":
            return 1
        if action == "use":
            log_success(f"Using existing params: {params_path}", indent=stage_indent)
            loaded = load_params(params_path)
            if not loaded:
                return 1
            tensor_types = loaded.tensor_types
            default_type = loaded.default_type
            imatrix = loaded.imatrix
            if not stem_label:
                quant_label = default_type
        else:
            if params_path.exists() and not remove_files([params_path]):
                return 1
            save_params_payload(payload, params_path)
            log_success(f"Params saved to {params_path}", indent=stage_indent)
            tensor_types = payload.tensor_types
            default_type = payload.default_type
            imatrix = payload.imatrix

        stage_indent = log_stage(f"Quantizing target GGUF: {quant_label}")
        output_name = (
            f"{prefix}-{target_repo_slug}-{quant_label}"
            if prefix
            else f"{target_repo_slug}-{quant_label}"
        )
        output_path = quantized_dir / f"{output_name}.gguf"
        existing_splits = _collect_split_outputs(output_path, quant_label)
        action = confirm_overwrite(
            [output_path, *existing_splits],
            "quantized output",
            indent=stage_indent,
        )
        if action == "cancel":
            return 1
        if action == "use":
            if output_path.exists():
                log_success(
                    f"Using existing quantized output: {output_path}",
                    indent=stage_indent,
                )
            else:
                log_success(
                    f"Using existing quantized shards: {existing_splits[0].parent}",
                    indent=stage_indent,
                )
            continue
        if output_path.exists() and not remove_files([output_path]):
            return 1

        result = quantize_gguf(
            converted_gguf,
            output_path,
            tensor_types=tensor_types,
            default_type=default_type,
            imatrix=imatrix,
            llama_quantize=tools.llama_quantize,
            cwd=work_dir,
            indent=stage_indent,
        )
        if result != 0:
            return result

        log_success(f"Quantized GGUF saved to {output_path}", indent=stage_indent)
        if config.template_copy_metadata:
            stage_indent = log_stage(
                f"Copying template metadata to quantized GGUF: {output_path.name}"
            )
            result = copy_template_metadata(
                template_group[0],
                output_path,
                config.template_copy_metadata,
                indent=stage_indent,
            )
            if result != 0:
                return result

        if config.output_apply_metadata:
            stage_indent = log_stage(
                f"Applying metadata to quantized GGUF: {output_path.name}"
            )
            result = apply_metadata(
                output_path,
                config.output_apply_metadata,
                indent=stage_indent,
            )
            if result != 0:
                return result

        stage_indent = log_stage(f"Splitting quantized GGUF: {output_path.name}")
        needed = split_needed(output_path.stat().st_size, config.output_split)
        if needed is None:
            print(f"Invalid output split size: {config.output_split}")
            return 1
        if not needed:
            log_success(
                f"Skipping split; {output_path.name} is <= {config.output_split}",
                indent=stage_indent,
            )
            continue
        split_dir = quantized_dir / quant_label
        split_output = split_dir / output_path.stem
        existing_splits = _collect_split_outputs(output_path, quant_label)
        action = confirm_overwrite(existing_splits, "split output", indent=stage_indent)
        if action == "cancel":
            return 1
        if action == "use":
            log_success(
                f"Using existing split output: {split_output}", indent=stage_indent
            )
            if not remove_files([output_path]):
                return 1
            continue
        if existing_splits and not remove_files(existing_splits):
            return 1

        split_dir.mkdir(parents=True, exist_ok=True)
        result = split_gguf(
            output_path,
            split_output,
            max_size=config.output_split,
            llama_gguf_split=tools.llama_gguf_split,
            cwd=work_dir,
            indent=stage_indent,
        )
        if result != 0:
            return result
        if not remove_files([output_path]):
            return 1
        log_success(f"Split GGUF saved to {split_output}", indent=stage_indent)

    if config.template_copy_files:
        stage_indent = log_stage("Copying template files to quantized outputs")
        result = copy_template_files(
            resolved.template_snapshot,
            config.template_copy_files,
            quantized_dir,
            indent=stage_indent,
        )
        if result != 0:
            return result

    return 0
