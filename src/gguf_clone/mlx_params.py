"""Convert GGUF tensor-type recipes to MLX quantization templates.

Translates GGUF tensor names (blk.N.attn_q) to MLX module paths
(layers.N.self_attn.q_proj) and GGUF quant types (Q4_K) to bit widths.

The output format is a dict consumable by quantize_template.py's Matcher:
  {"default_bits": 4, "tensor_types": {"layers\\.\\d+\\.self_attn\\.q_proj": 4, ...}}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

GGUF_TO_BITS: dict[str, int | str] = {
    "Q2_K": 2,
    "Q3_K": 3,
    "Q4_K": 4,
    "Q5_K": 5,
    "Q6_K": 6,
    "Q8_0": 8,
    "IQ4_XS": 4,
    "MXFP4": "mxfp4",
}

BASE_COMPONENT_MAP: dict[str, str] = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down_exps": "mlp.switch_mlp.down_proj",
    "ffn_gate_exps": "mlp.switch_mlp.gate_proj",
    "ffn_up_exps": "mlp.switch_mlp.up_proj",
    "ffn_down_shexp": "mlp.shared_experts.down_proj",
    "ffn_gate_shexp": "mlp.shared_experts.gate_proj",
    "ffn_up_shexp": "mlp.shared_experts.up_proj",
    "ffn_gate_inp": "mlp.gate",
}

ARCH_OVERRIDES: dict[str, dict[str, str]] = {
    "deepseek_v3": {
        "attn_q_a": "self_attn.q_a_proj",
        "attn_q_b": "self_attn.q_b_proj",
        "attn_kv_a_mqa": "self_attn.kv_a_proj_with_mqa",
        "attn_k_b": "self_attn.embed_q",
        "attn_v_b": "self_attn.unembed_out",
    },
    "qwen3_next": {
        "attn_qkv": "linear_attn.in_proj_qkvz",
        # attn_gate fused into self_attn.q_proj (2x width, split at runtime)
        "attn_gate": "self_attn.q_proj",
        "ssm_ba": "linear_attn.in_proj_ba",
        "ssm_conv1d": "linear_attn.conv1d",
        "ssm_out": "linear_attn.out_proj",
    },
    "deepseek_v32": {
        "attn_q_a": "self_attn.q_a_proj",
        "attn_q_b": "self_attn.q_b_proj",
        "attn_kv_a_mqa": "self_attn.kv_a_proj_with_mqa",
        "attn_k_b": "self_attn.embed_q",
        "attn_v_b": "self_attn.unembed_out",
        "indexer.attn_q_b": "self_attn.indexer.wq_b",
        "indexer.attn_k": "self_attn.indexer.wk",
        "indexer.proj": "self_attn.indexer.weights_proj",
    },
}

TOP_LEVEL_MAP: dict[str, str] = {
    "output": "lm_head",
    "token_embd": "embed_tokens",
}


def build_component_map(arch: str | None = None) -> dict[str, str]:
    cmap = dict(BASE_COMPONENT_MAP)
    if arch and arch in ARCH_OVERRIDES:
        cmap.update(ARCH_OVERRIDES[arch])
    return cmap


def convert_gguf_type(gguf_type: str) -> int | str:
    """Q4_K -> 4, F32 -> 'float32', etc."""
    t = gguf_type.strip().upper()
    if t == "F32":
        return "float32"
    if t == "F16":
        return "float16"
    if t in GGUF_TO_BITS:
        return GGUF_TO_BITS[t]
    m = re.match(r"^Q(\d+)", t)
    if m:
        bits = int(m.group(1))
        if 2 <= bits <= 8:
            return bits
    raise ValueError(f"Unknown GGUF type: {gguf_type}")


def convert_pattern(pattern: str, component_map: dict[str, str]) -> str | None:
    """Convert a GGUF regex pattern to an MLX module-path regex.

    Handles two forms:
      - Top-level: output.weight, token_embd.weight
      - Per-layer: blk\\.(layer_group)\\.component.weight

    Returns a pattern suitable for matching against MLX module paths
    (the .weight suffix is stripped).
    """
    for gguf_name, mlx_name in TOP_LEVEL_MAP.items():
        if re.match(rf"^{re.escape(gguf_name)}[\\]?\.weight$", pattern):
            return mlx_name

    if not pattern.startswith("blk"):
        return None

    result = "layers" + pattern[3:]

    # Longest-first to avoid partial matches (attn_q vs attn_qkv)
    matched = False
    for comp in sorted(component_map, key=len, reverse=True):
        idx = result.find(comp)
        if idx < 0:
            continue
        end = idx + len(comp)
        if end < len(result) and result[end] not in (".", "\\"):
            continue
        mlx_comp = component_map[comp].replace(".", "\\.")
        result = result[:idx] + mlx_comp + result[idx + len(comp) :]
        matched = True
        break

    if not matched:
        return None

    for suffix in ("\\.weight", ".weight"):
        if result.endswith(suffix):
            result = result[: -len(suffix)]
            break

    return result


@dataclass(frozen=True)
class MlxParams:
    default_bits: int
    tensor_types: dict[str, int | str]
    warnings: list[str] = field(default_factory=list)


def convert_gguf_params(
    tensor_types: list[str],
    default_type: str,
    arch: str | None = None,
) -> MlxParams:
    """Convert GGUF params (tensor_types list + default_type) to MLX format."""
    component_map = build_component_map(arch)
    warnings: list[str] = []
    mlx_types: dict[str, int | str] = {}

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

        mlx_pattern = convert_pattern(pattern.strip(), component_map)
        if mlx_pattern is None:
            warnings.append(f"Unmapped GGUF pattern: {pattern.strip()}")
            continue

        mlx_types[mlx_pattern] = value

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


def read_gguf_arch(gguf_path: Path) -> str | None:
    """Read general.architecture from a GGUF file's metadata."""
    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path), "r")
    arch_field = reader.fields.get("general.architecture")
    if arch_field is None:
        return None
    parts = arch_field.parts
    data_idxs = arch_field.data
    if not len(data_idxs):
        return None
    return bytes(parts[data_idxs[0]][data_idxs]).decode("utf-8", errors="replace")


def resolve_arch(mlx_arch: str | None, gguf_path: Path) -> str | None:
    """Resolve the MLX architecture string.

    'auto' reads from the GGUF metadata. None or explicit string passed through.
    """
    if mlx_arch is None:
        return None
    if mlx_arch == "auto":
        return read_gguf_arch(gguf_path)
    return mlx_arch


def save_mlx_params(params: MlxParams, output_path: Path) -> None:
    payload: dict[str, object] = {
        "default_bits": params.default_bits,
        "tensor_types": params.tensor_types,
    }
    _ = output_path.write_text(json.dumps(payload, indent=2))


def load_mlx_params(path: Path) -> MlxParams | None:
    try:
        raw = cast(object, json.loads(path.read_text()))
    except Exception as exc:
        print(f"Failed to read MLX params: {exc}")
        return None

    from .common import require_mapping

    data = require_mapping(raw, "MLX params file must be a JSON object.")
    if data is None:
        return None

    default_bits = data.get("default_bits")
    if not isinstance(default_bits, int):
        print("MLX params file is missing a valid 'default_bits'.")
        return None

    tensor_types_raw = data.get("tensor_types")
    if not isinstance(tensor_types_raw, dict):
        print("MLX params file is missing a valid 'tensor_types' dict.")
        return None

    typed_types: dict[str, int | str] = {}
    for k, v in cast(dict[object, object], tensor_types_raw).items():
        if not isinstance(k, str) or not isinstance(v, (int, str)):
            print(f"Invalid tensor_types entry: {k}={v}")
            return None
        typed_types[k] = v

    return MlxParams(default_bits=default_bits, tensor_types=typed_types)
