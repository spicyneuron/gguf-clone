"""MLX quantization params: type conversion, serialization, and loading.

Converts GGUF quant types (Q4_K) to MLX bit widths. Name mapping is
handled by mlx_discover.py.
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


@dataclass(frozen=True)
class MlxParams:
    default_bits: int
    tensor_types: dict[str, int | str]
    warnings: list[str] = field(default_factory=list)


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
