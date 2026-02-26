from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, cast

from gguf import GGMLQuantizationType, GGUFReader, ReaderTensor

from .common import require_mapping

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
    template_metadata: dict[str, str] = field(default_factory=dict)
    staged_files: list[str] = field(default_factory=list)
    template_gguf: str | None = None


class ParamsPayloadData(TypedDict):
    tensor_types: list[str]
    default_type: str
    imatrix: str
    template_metadata: dict[str, str]
    staged_files: list[str]
    template_gguf: str | None


def ignore_tensor(tensor: ReaderTensor) -> bool:
    if not tensor.name.endswith("weight"):
        return True
    if len(tensor.shape) < 2:
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
    # (prefix, suffix) -> {layer_number: qtype}
    layered: dict[tuple[str, str], dict[str, str]] = {}
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
                group = layered.setdefault((pre, post), {})
                existing = group.get(num)
                if existing is not None and existing != qtype:
                    name = f"{pre}.{num}.{post}"
                    raise ValueError(
                        f"Conflicting types for {name}: {existing} vs {qtype}"
                    )
                group[num] = qtype
            else:
                entry = f"{tensor.name}={qtype}"
                if entry not in seen_tensor_types:
                    tensor_types.append(entry)
                    seen_tensor_types.add(entry)

    for (pre, post), layers in layered.items():
        qtypes = set(layers.values())
        if len(qtypes) == 1:
            # All layers agree -- single wildcard entry
            qtype = next(iter(qtypes))
            tensor_types.append(f"{pre}\\.(\\d+)\\.{post}={qtype}")
        else:
            # Layers differ -- group by qtype with alternation
            by_qtype: dict[str, list[str]] = {}
            for num, qtype in layers.items():
                by_qtype.setdefault(qtype, []).append(num)
            for qtype, nums in by_qtype.items():
                group = "|".join(sorted(nums, key=int))
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


def save_params_payload(payload: ParamsPayload, output_path: Path) -> None:
    payload_dict = {
        "imatrix": payload.imatrix,
        "tensor_types": payload.tensor_types,
        "default_type": payload.default_type,
        "template_metadata": payload.template_metadata,
        "staged_files": payload.staged_files,
        "template_gguf": payload.template_gguf,
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

    template_metadata_raw = root.get("template_metadata", {})
    if not isinstance(template_metadata_raw, dict):
        print("Params file is missing a valid 'template_metadata' dict.")
        return None
    template_metadata: dict[str, str] = {}
    for key, value in cast(dict[object, object], template_metadata_raw).items():
        if not isinstance(key, str) or not isinstance(value, str):
            print("Params file is missing a valid 'template_metadata' dict.")
            return None
        template_metadata[key] = value

    staged_files_raw = root.get("staged_files", [])
    if not isinstance(staged_files_raw, list):
        print("Params file is missing a valid 'staged_files' list.")
        return None
    staged_files: list[str] = []
    for item in cast(list[object], staged_files_raw):
        if not isinstance(item, str):
            print("Params file is missing a valid 'staged_files' list.")
            return None
        staged_files.append(item)

    template_gguf = root.get("template_gguf", None)
    if template_gguf is not None and not isinstance(template_gguf, str):
        print("Params file is missing a valid 'template_gguf' value.")
        return None

    return {
        "tensor_types": tensor_types,
        "default_type": default_type,
        "imatrix": imatrix,
        "template_metadata": template_metadata,
        "staged_files": staged_files,
        "template_gguf": template_gguf,
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
        template_metadata=parsed["template_metadata"],
        staged_files=parsed["staged_files"],
        template_gguf=parsed["template_gguf"],
    )
