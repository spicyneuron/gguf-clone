from __future__ import annotations

from pathlib import Path

from .. import config as config_mod
from ..artifacts import Artifacts
from ..common import confirm_overwrite, log_stage, log_success, remove_files
from ..convert import convert_target
from ..files import copy_staged_files
from ..metadata import apply_metadata
from ..params import load_params
from ..quantize import quantize_gguf
from ..resolve import (
    ModelResolutionError,
    ToolResolutionError,
    print_env_hint,
    resolve_source_snapshot,
    resolve_tools,
)
from ..split import split_gguf, split_needed


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


def _resolve_converted_gguf(
    *,
    config: config_mod.RunConfig,
    artifacts: Artifacts,
    convert_script: Path,
    work_dir: Path,
) -> Path | None:
    qg = config.quantize_gguf
    if qg is None:
        print("quantize_gguf section missing from config.")
        return None

    if qg.target_gguf:
        target_gguf_path = Path(qg.target_gguf).expanduser()
        if not target_gguf_path.is_absolute():
            target_gguf_path = (work_dir / target_gguf_path).resolve()
        if not target_gguf_path.exists():
            print(f"target_gguf not found: {target_gguf_path}")
            return None
        stage_indent = log_stage("Using explicit target GGUF")
        log_success(f"Using target GGUF: {target_gguf_path}", indent=stage_indent)
        return target_gguf_path

    if not qg.target_convert:
        print("No target_gguf and target_convert is false.")
        return None

    try:
        target_snapshot = resolve_source_snapshot(config.target)
    except ModelResolutionError as exc:
        print(str(exc))
        return None

    stage_indent = log_stage("Converting target model to GGUF")
    result = convert_target(
        target_snapshot,
        output_dir=artifacts.converted_dir,
        outfile_name=artifacts.converted_gguf().name,
        existing_glob=artifacts.converted_gguf().name,
        convert_script=convert_script,
        cwd=work_dir,
        indent=stage_indent,
        use_message="Using existing converted output.",
    )
    if result != 0:
        return None

    converted_gguf = artifacts.converted_gguf()
    if not converted_gguf.exists():
        print(f"Converted GGUF not found: {converted_gguf}")
        return None
    return converted_gguf


def _list_params_files(artifacts: Artifacts) -> list[Path]:
    params_suffix = "-gguf.json"
    params_prefix = f"{artifacts.template_slug}-"
    return sorted(
        path
        for path in artifacts.params_dir.iterdir()
        if path.name.startswith(params_prefix) and path.name.endswith(params_suffix)
    )


def run_quantize_gguf_stage(
    config: config_mod.RunConfig,
    artifacts: Artifacts,
) -> int:
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

    converted_gguf = _resolve_converted_gguf(
        config=config,
        artifacts=artifacts,
        convert_script=tools.convert_hf_to_gguf,
        work_dir=work_dir,
    )
    if converted_gguf is None:
        return 1

    params_files = _list_params_files(artifacts)
    if not params_files:
        print(f"No -gguf.json params files found in {artifacts.params_dir}")
        return 1

    params_prefix = f"{artifacts.template_slug}-"
    params_suffix = "-gguf.json"
    staged_files: set[str] = set()

    for params_path in params_files:
        quant_label = params_path.name[len(params_prefix) : -len(params_suffix)]

        loaded = load_params(params_path)
        if not loaded:
            return 1

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
        if existing_splits and not remove_files(existing_splits):
            return 1

        result = quantize_gguf(
            converted_gguf,
            output_path,
            tensor_types=loaded.tensor_types,
            default_type=loaded.default_type,
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
                output_path,
                extracted_metadata,
                indent=stage_indent,
            )
            if result != 0:
                return result

        if qg.apply_metadata:
            stage_indent = log_stage(f"Applying metadata: {output_path.name}")
            result = apply_metadata(
                output_path,
                qg.apply_metadata,
                indent=stage_indent,
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
            existing_splits,
            "split output",
            indent=stage_indent,
        )
        if action == "cancel":
            return 1
        if action == "use":
            log_success(f"Using existing split: {split_output}", indent=stage_indent)
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
