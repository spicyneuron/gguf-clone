"""Stage: extract-template-mlx

Translates GGUF params into MLX quantization configs by introspecting
the target MLX model architecture at runtime.
"""

from __future__ import annotations

import re

from .. import config as config_mod
from ..artifacts import Artifacts
from ..common import confirm_overwrite, log_success, remove_files
from ..mlx_discover import (
    MlxMapping,
    build_mapping,
    extract_gguf_components,
    get_mlx_param_paths,
)
from ..mlx_params import MlxParams, convert_gguf_type, save_mlx_params
from ..params import load_params
from ..resolve import ModelResolutionError, resolve_source_snapshot


def _convert_tensor_types(
    tensor_types: list[str],
    default_type: str,
    mapping: MlxMapping,
) -> MlxParams:
    """Convert GGUF tensor_types to MLX format using a discovered mapping."""
    warnings: list[str] = list(mapping.warnings)
    mlx_types: dict[str, int | str] = {}
    cmap = mapping.component_map

    for entry in tensor_types:
        if "=" not in entry:
            warnings.append(f"Skipping malformed entry: {entry}")
            continue
        pattern, gguf_type = entry.rsplit("=", 1)

        try:
            value = convert_gguf_type(gguf_type)
        except ValueError as e:
            warnings.append(f"Skipping unknown type: {entry} ({e})")
            continue

        clean = pattern.strip().replace("\\.", ".")

        # Top-level patterns: output.weight, token_embd.weight
        m = re.match(r"^([a-z_]+)\.weight$", clean)
        if m:
            gguf_comp = m.group(1)
            mlx_comp = cmap.get(gguf_comp)
            if mlx_comp is None:
                warnings.append(f"Unmapped GGUF pattern: {pattern.strip()}")
                continue
            mlx_types[mlx_comp] = value
            continue

        # Per-layer patterns: blk.(group).component.weight
        m = re.match(r"^blk\.(?:\((.+?)\)|\d+)\.(.+?)\.weight$", clean)
        if m:
            layer_group = m.group(1) or r"\d+"
            gguf_comp = m.group(2)
            mlx_comp = cmap.get(gguf_comp)
            if mlx_comp is None:
                warnings.append(f"Unmapped GGUF pattern: {pattern.strip()}")
                continue
            mlx_pattern = rf"layers\.({layer_group})\.{mlx_comp}"
            # Re-escape dots in the mlx component for regex matching
            mlx_pattern = "layers" + rf"\.({layer_group})\." + mlx_comp.replace(".", "\\.")
            mlx_types[mlx_pattern] = value
            continue

        warnings.append(f"Unmapped GGUF pattern: {pattern.strip()}")

    try:
        default_bits = convert_gguf_type(default_type)
    except ValueError:
        default_bits = 4
        warnings.append(f"Unknown default_type '{default_type}', using 4")

    if isinstance(default_bits, str):
        warnings.append(
            f"default_type '{default_type}' is a float dtype, using 4 bits"
        )
        default_bits = 4

    return MlxParams(
        default_bits=default_bits,
        tensor_types=mlx_types,
        warnings=warnings,
    )


def run_extract_template_mlx_stage(
    config: config_mod.RunConfig,
    artifacts: Artifacts,
) -> int:
    epm = config.extract_template_mlx
    if epm is None:
        print("extract_template_mlx section missing from config.")
        return 1

    # Check mlx-lm availability early
    try:
        import importlib

        _ = importlib.import_module("mlx_lm")
    except ImportError:
        print("mlx-lm is required for extract-template-mlx. Install with: uv sync --group mlx")
        return 1

    artifacts.mkdir_all()

    # Find GGUF params files from extract-template stage
    params_files = sorted(artifacts.params_dir.glob("*-gguf.json"))
    if not params_files:
        print("No GGUF params files found. Run extract-template first.")
        return 1

    # Resolve target model config (just JSON files, no weights)
    try:
        target_path = resolve_source_snapshot(
            config.target, allow_patterns=["*.json"]
        )
    except ModelResolutionError as exc:
        print(str(exc))
        return 1

    # Discover MLX param tree
    try:
        mlx_paths = get_mlx_param_paths(target_path, epm.trust_remote_code)
    except ImportError as exc:
        print(str(exc))
        return 1
    except Exception as exc:
        print(f"Failed to discover MLX model structure: {exc}")
        return 1

    for params_file in params_files:
        payload = load_params(params_file)
        if payload is None:
            print(f"Failed to load params from {params_file}")
            return 1

        # Extract quant label from filename: {slug}-{label}-gguf.json
        stem = params_file.stem  # slug-LABEL-gguf
        if stem.endswith("-gguf"):
            stem = stem[: -len("-gguf")]
        # Label is everything after the template slug
        slug = artifacts.template_slug
        if stem.startswith(slug + "-"):
            quant_label = stem[len(slug) + 1 :]
        else:
            quant_label = stem

        mlx_path = artifacts.params_mlx(quant_label)
        action = confirm_overwrite([mlx_path], "MLX params", indent="")
        if action == "cancel":
            return 1
        if action == "use":
            log_success(f"Using existing params: {mlx_path}")
            continue

        if mlx_path.exists() and not remove_files([mlx_path]):
            return 1

        gguf_components = extract_gguf_components(payload.tensor_types)
        mapping = build_mapping(gguf_components, mlx_paths)

        mlx_params = _convert_tensor_types(
            payload.tensor_types, payload.default_type, mapping
        )
        for warning in mlx_params.warnings:
            print(f"  [WARN] {warning}")

        save_mlx_params(mlx_params, mlx_path)
        log_success(f"MLX params saved to {mlx_path}")

    return 0
