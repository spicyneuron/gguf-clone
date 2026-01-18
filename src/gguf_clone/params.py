from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from gguf import GGMLQuantizationType, GGUFReader, ReaderTensor

from .common import require_mapping

# Adapted from https://github.com/unslothai/llama.cpp/blob/master/src/llama-quant.cpp
IGNORE = [
    "_norm.weight",
    "ffn_gate_inp.weight",
    "altup",
    "laurel",
    "per_layer_model_proj",
    "position_embd.weight",
    "pos_embd.weight",
    "token_type_embd.weight",
    "token_types.weight",
    "ssm_conv1d.weight",
    "shortconv.conv.weight",
    "time_mix_first.weight",
    "time_mix_w0.weight",
    "time_mix_w1.weight",
    "time_mix_w2.weight",
    "time_mix_v0.weight",
    "time_mix_v1.weight",
    "time_mix_v2.weight",
    "time_mix_a0.weight",
    "time_mix_a1.weight",
    "time_mix_a2.weight",
    "time_mix_g1.weight",
    "time_mix_g2.weight",
    "time_mix_decay_w1.weight",
    "time_mix_decay_w2.weight",
    "time_mix_lerp_fused.weight",
    "attn_rel_b.weight",
    ".position_embd.",
]


LAYER_NAME = re.compile(r"^(.*?)\.(\d+)\.(.*)$")


@dataclass(frozen=True)
class QuantParams:
    tensor_types: list[str]
    default_type: str
    quant_type_counts: dict[str, int]


@dataclass(frozen=True)
class ParamsPayload:
    tensor_types: list[str]
    default_type: str
    imatrix: str


class ParamsPayloadData(TypedDict):
    tensor_types: list[str]
    default_type: str
    imatrix: str


def ignore_tensor(tensor: ReaderTensor) -> bool:
    if not tensor.name.endswith("weight"):
        return True
    if len(tensor.shape) < 2:
        return True
    if any(s in tensor.name for s in IGNORE):
        return True
    return False


def _normalize_paths(paths: Path | list[Path]) -> list[Path]:
    if isinstance(paths, Path):
        return [paths]
    return list(paths)


def copy_imatrix(src: Path, dest_dir: Path, prefix: str = "") -> str:
    """Copy imatrix file to dest_dir and return a relative path.

    Returns a relative path (prefix + filename) so that GGUF metadata
    doesn't contain absolute paths from the build machine.
    """
    dest = dest_dir / src.name
    if not (dest.exists() and dest.samefile(src)):
        _ = shutil.copy2(src, dest)
    if prefix:
        return f"{prefix}/{src.name}"
    return src.name


def build_params(paths: Path | list[Path]) -> QuantParams:
    tensor_types: list[str] = []
    seen_tensor_types: set[str] = set()
    seen_tensors: dict[tuple[str, str, str], set[str]] = {}
    quant_type_counts: dict[str, int] = {}

    for path in _normalize_paths(paths):
        reader = GGUFReader(str(path), "r")
        for tensor in reader.tensors:
            if ignore_tensor(tensor):
                continue
            match = LAYER_NAME.match(tensor.name)
            qtype = GGMLQuantizationType(tensor.tensor_type).name
            quant_type_counts[qtype] = quant_type_counts.get(qtype, 0) + 1
            if match:
                pre, num, post = match.group(1), match.group(2), match.group(3)
                seen_tensors.setdefault((pre, post, qtype), set()).add(num)
            else:
                entry = f"{tensor.name}={qtype}"
                if entry not in seen_tensor_types:
                    tensor_types.append(entry)
                    seen_tensor_types.add(entry)

    for (pre, post, qtype), numbers in seen_tensors.items():
        group = "|".join(sorted(numbers, key=int))
        tensor_types.append(f"{pre}\\.({group})\\.{post}={qtype}")

    default_type = (
        max(quant_type_counts, key=quant_type_counts.__getitem__)
        if quant_type_counts
        else "Q8_0"
    )

    return QuantParams(
        tensor_types=tensor_types,
        default_type=default_type,
        quant_type_counts=quant_type_counts,
    )


def build_params_payload(
    template_paths: Path | list[Path],
    imatrix: str,
) -> tuple[ParamsPayload, QuantParams]:
    params = build_params(template_paths)
    payload = ParamsPayload(
        tensor_types=params.tensor_types,
        default_type=params.default_type,
        imatrix=imatrix,
    )
    return payload, params


def save_params_payload(payload: ParamsPayload, output_path: Path) -> None:
    payload_dict = {
        "imatrix": payload.imatrix,
        "tensor_types": payload.tensor_types,
        "default_type": payload.default_type,
    }
    _ = output_path.write_text(json.dumps(payload_dict, indent=2))


def _load_json(text: str) -> object:
    return cast(object, json.loads(text))


def _parse_tensor_types(value: object) -> list[str] | None:
    if not isinstance(value, list):
        print("Params file is missing a valid 'tensor_types' list.")
        return None
    items = cast(list[object], value)
    cleaned: list[str] = []
    for item in items:
        if not isinstance(item, str):
            print("Params file is missing a valid 'tensor_types' list.")
            return None
        cleaned.append(item)
    return cleaned


def _parse_params_payload(data: object) -> ParamsPayloadData | None:
    root = require_mapping(data, "Params file must be a JSON object.")
    if root is None:
        return None

    default_type = root.get("default_type")
    if not isinstance(default_type, str):
        print("Params file is missing a valid 'default_type'.")
        return None

    tensor_types = _parse_tensor_types(root.get("tensor_types"))
    if tensor_types is None:
        return None

    imatrix = root.get("imatrix", "")
    if not isinstance(imatrix, str):
        print("Params file is missing a valid 'imatrix' path.")
        return None

    return {
        "tensor_types": tensor_types,
        "default_type": default_type,
        "imatrix": imatrix,
    }


def load_params(path: Path) -> ParamsPayload | None:
    try:
        payload = _load_json(path.read_text())
    except Exception as exc:
        print(f"Failed to read params: {exc}")
        return None

    parsed = _parse_params_payload(payload)
    if not parsed:
        return None

    return ParamsPayload(
        tensor_types=parsed["tensor_types"],
        default_type=parsed["default_type"],
        imatrix=parsed["imatrix"],
    )
