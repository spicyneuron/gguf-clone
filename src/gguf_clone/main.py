from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from . import config as config_mod
from .artifacts import Artifacts, quant_label_from_stem as _v2_quant_label
from .common import (
    OverwriteBehavior,
    confirm_overwrite,
    log_stage,
    log_success,
    remove_files,
    set_overwrite_behavior,
    set_verbose,
)
from .convert import convert_target
from .files import copy_staged_files, copy_template_files, match_copy_files
from .metadata import apply_metadata, extract_template_metadata
from .mlx_params import convert_gguf_params, resolve_arch, save_mlx_params
from .params import (
    ParamsPayload,
    build_params,
    copy_imatrix,
    load_params,
    save_params_payload,
)
from .quantize import quantize_gguf
from .resolve import (
    ModelResolutionError,
    ToolResolutionError,
    check_deps,
    check_gguf_support,
    match_pattern,
    print_env_hint,
    resolve_source_snapshot,
    resolve_tools,
)
from .split import split_gguf, split_needed


def _init_run(
    *,
    verbose: bool,
    overwrite_behavior: OverwriteBehavior | None,
) -> str | None:
    """Shared init for v2 stages. Returns an error message or None."""
    set_verbose(verbose)
    set_overwrite_behavior(overwrite_behavior)
    missing = check_deps()
    if missing:
        return "Missing Python dependencies: " + ", ".join(missing)
    gguf_error = check_gguf_support()
    if gguf_error:
        return gguf_error
    return None


def _resolve_template_snapshot(
    ref: config_mod.SourceRef,
    allow_patterns: list[str],
) -> Path | None:
    try:
        return resolve_source_snapshot(ref, allow_patterns=allow_patterns or None)
    except ModelResolutionError as exc:
        print(str(exc))
        return None


def _imatrix_exists(imatrix: str, *, work_dir: Path) -> bool:
    if not imatrix:
        return True
    imatrix_path = Path(imatrix)
    if not imatrix_path.is_absolute():
        imatrix_path = work_dir / imatrix_path
    return imatrix_path.exists()


def _collect_gguf_split_outputs(output_path: Path, quant_label: str) -> list[Path]:
    split_dir = output_path.parent / quant_label
    if not split_dir.exists():
        return []
    return sorted(split_dir.glob(f"{output_path.stem}-*-of-*.gguf"))


def run_extract_params(
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

    ep = config.extract_params
    if ep is None:
        print("extract_params section missing from config.")
        return 1

    artifacts.mkdir_all()

    qg = config.quantize_gguf
    template_allow = list(ep.ggufs)
    if qg is not None:
        if qg.imatrix:
            template_allow.append(qg.imatrix)
        if qg.copy_files:
            template_allow.extend(qg.copy_files)
        if qg.copy_metadata:
            template_allow.append("*.gguf")

    template_snapshot = _resolve_template_snapshot(config.template, template_allow)
    if template_snapshot is None:
        return 1

    gguf_candidates = sorted(
        p for p in template_snapshot.rglob("*.gguf") if p.is_file()
    )
    if not gguf_candidates:
        print(f"No .gguf files found in template: {template_snapshot}")
        return 1

    imatrix_rel = ""
    if qg is not None and qg.imatrix:
        imatrix_candidates = [
            p for p in template_snapshot.rglob("*") if p.is_file()
        ]
        imatrix_matches = match_pattern(
            template_snapshot,
            imatrix_candidates,
            qg.imatrix,
            "imatrix",
            require_single=True,
        )
        if not imatrix_matches:
            return 1
        imatrix_rel = copy_imatrix(
            imatrix_matches[0], artifacts.params_dir, prefix="params"
        )

    staged_file_names: list[str] = []
    if qg is not None and qg.copy_files:
        matched_files = match_copy_files(template_snapshot, qg.copy_files)
        if matched_files is None:
            return 1
        staged_file_names = sorted({path.name for path in matched_files})

        stage_indent = log_stage("Staging template files for later stages")
        result = copy_template_files(
            template_snapshot,
            qg.copy_files,
            artifacts.template_files_dir,
            indent=stage_indent,
        )
        if result != 0:
            return result

    for gguf_pattern in ep.ggufs:
        template_group = match_pattern(
            template_snapshot, gguf_candidates, gguf_pattern, "template GGUF"
        )
        if not template_group:
            return 1

        stem_labels = {
            label
            for path in template_group
            if (label := _v2_quant_label(path.stem))
        }
        if len(stem_labels) > 1:
            print("Template GGUF pattern matched multiple quant labels:")
            for label in sorted(stem_labels):
                print(f"  {label}")
            return 1

        params = build_params(template_group)
        quant_label = next(iter(stem_labels), None) or params.default_type
        template_gguf = template_group[0].relative_to(template_snapshot).as_posix()

        template_metadata: dict[str, str] = {}
        if qg is not None and qg.copy_metadata:
            extracted = extract_template_metadata(
                template_group[0],
                qg.copy_metadata,
            )
            if extracted is None:
                return 1
            template_metadata = extracted

        if "gguf" in ep.targets:
            gguf_path = artifacts.params_gguf(quant_label)
            action = confirm_overwrite([gguf_path], "params", indent="")
            if action == "cancel":
                return 1
            if action == "use":
                log_success(f"Using existing params: {gguf_path}")
            else:
                if gguf_path.exists() and not remove_files([gguf_path]):
                    return 1
                payload = ParamsPayload(
                    tensor_types=params.tensor_types,
                    default_type=params.default_type,
                    imatrix=imatrix_rel,
                    template_metadata=template_metadata,
                    staged_files=staged_file_names,
                    template_gguf=template_gguf,
                )
                save_params_payload(payload, gguf_path)
                log_success(f"Params saved to {gguf_path}")

        if "mlx" in ep.targets:
            mlx_path = artifacts.params_mlx(quant_label)
            action = confirm_overwrite([mlx_path], "params", indent="")
            if action == "cancel":
                return 1
            if action == "use":
                log_success(f"Using existing params: {mlx_path}")
            else:
                if mlx_path.exists() and not remove_files([mlx_path]):
                    return 1
                arch = resolve_arch(ep.mlx_arch, template_group[0])
                mlx_params = convert_gguf_params(
                    params.tensor_types, params.default_type, arch=arch
                )
                for warning in mlx_params.warnings:
                    print(f"  [WARN] {warning}")
                save_mlx_params(mlx_params, mlx_path)
                log_success(f"MLX params saved to {mlx_path}")

    return 0


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

    qg = config.quantize_gguf
    if qg is None:
        print("quantize_gguf section missing from config.")
        return 1

    try:
        tools = resolve_tools()
    except ToolResolutionError as exc:
        print(str(exc))
        print_env_hint()
        return 1

    artifacts.mkdir_all()
    work_dir = artifacts.output_dir

    converted_gguf: Path
    if qg.target_gguf:
        target_gguf_path = Path(qg.target_gguf).expanduser()
        if not target_gguf_path.is_absolute():
            target_gguf_path = (work_dir / target_gguf_path).resolve()
        if not target_gguf_path.exists():
            print(f"target_gguf not found: {target_gguf_path}")
            return 1
        converted_gguf = target_gguf_path
        stage_indent = log_stage("Using explicit target GGUF")
        log_success(f"Using target GGUF: {converted_gguf}", indent=stage_indent)
    elif qg.target_convert:
        try:
            target_snapshot = resolve_source_snapshot(config.target)
        except ModelResolutionError as exc:
            print(str(exc))
            return 1
        stage_indent = log_stage("Converting target model to GGUF")
        result = convert_target(
            target_snapshot,
            output_dir=artifacts.converted_dir,
            outfile_name=artifacts.converted_gguf().name,
            existing_glob=artifacts.converted_gguf().name,
            convert_script=tools.convert_hf_to_gguf,
            cwd=work_dir,
            indent=stage_indent,
            use_message="Using existing converted output.",
        )
        if result != 0:
            return result
        converted_gguf = artifacts.converted_gguf()
        if not converted_gguf.exists():
            print(f"Converted GGUF not found: {converted_gguf}")
            return 1
    else:
        print("No target_gguf and target_convert is false.")
        return 1

    params_suffix = "-gguf.json"
    params_prefix = f"{artifacts.template_slug}-"
    params_files = sorted(
        p
        for p in artifacts.params_dir.iterdir()
        if p.name.startswith(params_prefix) and p.name.endswith(params_suffix)
    )
    if not params_files:
        print(f"No -gguf.json params files found in {artifacts.params_dir}")
        return 1

    staged_files: set[str] = set()

    for params_path in params_files:
        name_body = params_path.name[len(params_prefix) : -len(params_suffix)]
        quant_label = name_body

        loaded = load_params(params_path)
        if not loaded:
            return 1
        tensor_types = loaded.tensor_types
        default_type = loaded.default_type

        imatrix = ""
        if qg.imatrix:
            imatrix = loaded.imatrix
            if not imatrix:
                print(
                    f"Params file missing imatrix path: {params_path}. "
                    + "Run extract-params to stage template artifacts."
                )
                return 1
            if not _imatrix_exists(imatrix, work_dir=work_dir):
                print(f"imatrix file not found: {imatrix}")
                return 1

        extracted_metadata: dict[str, str] = {}
        if qg.copy_metadata:
            missing_metadata = [
                key for key in qg.copy_metadata if key not in loaded.template_metadata
            ]
            if missing_metadata:
                print(
                    "Params file missing extracted metadata keys: "
                    + ", ".join(missing_metadata)
                )
                print(f"  {params_path}")
                print("Run extract-params to stage template metadata.")
                return 1
            extracted_metadata = {
                key: loaded.template_metadata[key] for key in qg.copy_metadata
            }

        if qg.copy_files:
            if not loaded.staged_files:
                print(
                    f"Params file missing staged_files: {params_path}. "
                    + "Run extract-params to stage template files."
                )
                return 1
            staged_files.update(loaded.staged_files)

        stage_indent = log_stage(f"Quantizing target GGUF: {quant_label}")
        output_path = artifacts.quantized_gguf(quant_label)
        existing_splits = _collect_gguf_split_outputs(output_path, quant_label)
        action = confirm_overwrite(
            [output_path, *existing_splits], "quantized output", indent=stage_indent
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
        if existing_splits and not remove_files(existing_splits):
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

        if extracted_metadata:
            stage_indent = log_stage(
                f"Applying extracted template metadata: {output_path.name}"
            )
            result = apply_metadata(
                output_path, extracted_metadata, indent=stage_indent
            )
            if result != 0:
                return result

        if qg.apply_metadata:
            stage_indent = log_stage(f"Applying metadata: {output_path.name}")
            result = apply_metadata(
                output_path, qg.apply_metadata, indent=stage_indent
            )
            if result != 0:
                return result

        stage_indent = log_stage(f"Splitting: {output_path.name}")
        needed = split_needed(output_path.stat().st_size, qg.output_max_size)
        if needed is None:
            print(f"Invalid output_max_size: {qg.output_max_size}")
            return 1
        if not needed:
            log_success(
                f"Skipping split; {output_path.name} is <= {qg.output_max_size}",
                indent=stage_indent,
            )
            continue

        split_dir = artifacts.quantized_gguf_dir / quant_label
        split_output = split_dir / output_path.stem
        existing_splits = _collect_gguf_split_outputs(output_path, quant_label)
        action = confirm_overwrite(
            existing_splits, "split output", indent=stage_indent
        )
        if action == "cancel":
            return 1
        if action == "use":
            log_success(
                f"Using existing split: {split_output}", indent=stage_indent
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
            max_size=qg.output_max_size,
            llama_gguf_split=tools.llama_gguf_split,
            cwd=work_dir,
            indent=stage_indent,
        )
        if result != 0:
            return result
        if not remove_files([output_path]):
            return 1
        log_success(f"Split GGUF saved to {split_output}", indent=stage_indent)

    if qg.copy_files:
        if not staged_files:
            print("No staged template files found in params. Run extract-params first.")
            return 1
        stage_indent = log_stage("Copying staged template files to quantized outputs")
        result = copy_staged_files(
            artifacts.template_files_dir,
            sorted(staged_files),
            artifacts.quantized_gguf_dir,
            indent=stage_indent,
        )
        if result != 0:
            return result

    return 0


def run_pipeline(
    config_path: Path,
    *,
    verbose: bool = False,
    overwrite_behavior: OverwriteBehavior | None = None,
) -> int:
    """Run declared v2 stages in order."""
    config_path = config_path.expanduser()
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return 1

    config = config_mod.load_config(config_path)
    if not config:
        return 1

    artifacts = Artifacts.from_config(config.output_dir, config.template, config.target)

    stages: list[tuple[object, Callable[..., int]]] = [
        (config.extract_params, run_extract_params),
        (config.quantize_gguf, run_quantize_gguf),
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
        print("No stages declared in config (extract_params, quantize_gguf, quantize_mlx).")
        return 1

    return 0


def run(
    config_path: Path,
    *,
    verbose: bool = False,
    overwrite_behavior: OverwriteBehavior | None = None,
) -> int:
    """Backward-compatible entrypoint: delegates to v2 pipeline."""
    return run_pipeline(
        config_path,
        verbose=verbose,
        overwrite_behavior=overwrite_behavior,
    )
