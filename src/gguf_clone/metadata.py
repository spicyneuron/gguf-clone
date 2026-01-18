from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import SupportsInt, cast

from gguf import GGUFReader, GGUFValueType, GGUFWriter, Keys

from .common import log_line, log_success, remove_files


@dataclass(frozen=True)
class MetadataValue:
    value: object
    value_type: GGUFValueType
    sub_type: GGUFValueType | None


def copy_template_metadata(
    template_path: Path,
    target_path: Path,
    keys: list[str],
    *,
    indent: str = "",
) -> int:
    if not keys:
        return 0

    copy_keys: list[str] = []
    skipped: list[str] = []
    for key in dict.fromkeys(keys):
        if key == Keys.General.ARCHITECTURE or key.startswith("GGUF."):
            skipped.append(key)
            continue
        copy_keys.append(key)

    if skipped:
        log_line(f"Skipping managed metadata keys: {', '.join(skipped)}", indent=indent)

    if not copy_keys:
        return 0

    log_line(f"Copying metadata: {', '.join(copy_keys)}", indent=indent)

    template_reader = GGUFReader(str(template_path), "r")
    target_reader = GGUFReader(str(target_path), "r")

    overrides: dict[str, MetadataValue] = {}
    missing: list[str] = []
    for key in copy_keys:
        field = template_reader.get_field(key)
        if field is None:
            missing.append(key)
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == GGUFValueType.ARRAY else None
        field_value = cast(object, field.contents())
        overrides[key] = MetadataValue(field_value, value_type, sub_type)

    if missing:
        print("Template GGUF missing metadata keys:")
        for key in missing:
            print(f"  {key}")
        return 1

    arch_field = target_reader.get_field(Keys.General.ARCHITECTURE)
    if arch_field is None:
        print(f"Quantized GGUF missing required metadata: {Keys.General.ARCHITECTURE}")
        return 1
    arch_value = cast(object, arch_field.contents())
    if not isinstance(arch_value, str) or not arch_value:
        print(f"Quantized GGUF metadata {Keys.General.ARCHITECTURE} must be a string.")
        return 1
    arch: str = arch_value

    temp_path = target_path.with_suffix(f"{target_path.suffix}.metadata")
    if temp_path.exists() and not remove_files([temp_path]):
        return 1

    writer = GGUFWriter(temp_path, arch=arch, endianess=target_reader.endianess)

    alignment_value: object | None = None
    if Keys.General.ALIGNMENT in overrides:
        alignment_value = overrides[Keys.General.ALIGNMENT].value
    else:
        alignment_field = target_reader.get_field(Keys.General.ALIGNMENT)
        if alignment_field is not None:
            alignment_value = cast(object, alignment_field.contents())
    if alignment_value is not None:
        try:
            alignment_candidate = cast(SupportsInt | str | bytes | bytearray, alignment_value)
            writer.data_alignment = int(alignment_candidate)
        except (TypeError, ValueError):
            pass

    remaining = dict(overrides)
    for field in target_reader.fields.values():
        if field.name == Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name in remaining:
            value = remaining.pop(field.name)
            writer.add_key_value(field.name, value.value, value.value_type, sub_type=value.sub_type)
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == GGUFValueType.ARRAY else None
        field_value = cast(object, field.contents())
        writer.add_key_value(field.name, field_value, value_type, sub_type=sub_type)

    for key, value in remaining.items():
        writer.add_key_value(key, value.value, value.value_type, sub_type=value.sub_type)

    for tensor in target_reader.tensors:
        writer.add_tensor_info(  # pyright: ignore[reportUnknownMemberType]
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for tensor in target_reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=target_reader.endianess)

    writer.close()
    del target_reader
    del template_reader

    _ = temp_path.replace(target_path)
    log_success(f"Metadata copied to {target_path}", indent=indent)
    return 0


def apply_metadata(
    target_path: Path,
    metadata: dict[str, str],
    *,
    indent: str = "",
) -> int:
    if not metadata:
        return 0

    apply_keys: list[str] = []
    skipped: list[str] = []
    for key in dict.fromkeys(metadata.keys()):
        if key == Keys.General.ARCHITECTURE or key.startswith("GGUF."):
            skipped.append(key)
            continue
        apply_keys.append(key)

    if skipped:
        log_line(f"Skipping managed metadata keys: {', '.join(skipped)}", indent=indent)

    if not apply_keys:
        return 0

    log_line(f"Applying metadata: {', '.join(apply_keys)}", indent=indent)

    target_reader = GGUFReader(str(target_path), "r")

    arch_field = target_reader.get_field(Keys.General.ARCHITECTURE)
    if arch_field is None:
        print(f"Target GGUF missing required metadata: {Keys.General.ARCHITECTURE}")
        return 1
    arch_value = cast(object, arch_field.contents())
    if not isinstance(arch_value, str) or not arch_value:
        print(f"Target GGUF metadata {Keys.General.ARCHITECTURE} must be a string.")
        return 1
    arch: str = arch_value

    temp_path = target_path.with_suffix(f"{target_path.suffix}.metadata")
    if temp_path.exists() and not remove_files([temp_path]):
        return 1

    writer = GGUFWriter(temp_path, arch=arch, endianess=target_reader.endianess)

    # Handle alignment
    alignment_field = target_reader.get_field(Keys.General.ALIGNMENT)
    if alignment_field is not None:
        alignment_value = cast(object, alignment_field.contents())
        if alignment_value is not None:
            try:
                alignment_candidate = cast(SupportsInt | str | bytes | bytearray, alignment_value)
                writer.data_alignment = int(alignment_candidate)
            except (TypeError, ValueError):
                pass

    # Build override dict with string values
    overrides: dict[str, MetadataValue] = {}
    for key in apply_keys:
        value = metadata[key]
        overrides[key] = MetadataValue(value, GGUFValueType.STRING, None)

    # Write metadata: first from target, then apply overrides, then add new keys
    remaining = dict(overrides)
    for field in target_reader.fields.values():
        if field.name == Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name in remaining:
            value = remaining.pop(field.name)
            writer.add_key_value(field.name, value.value, value.value_type, sub_type=value.sub_type)
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == GGUFValueType.ARRAY else None
        field_value = cast(object, field.contents())
        writer.add_key_value(field.name, field_value, value_type, sub_type=sub_type)

    # Add any new keys that weren't in the original file
    for key, value in remaining.items():
        writer.add_key_value(key, value.value, value.value_type, sub_type=value.sub_type)

    # Copy tensor info and data
    for tensor in target_reader.tensors:
        writer.add_tensor_info(  # pyright: ignore[reportUnknownMemberType]
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for tensor in target_reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=target_reader.endianess)

    writer.close()
    del target_reader

    _ = temp_path.replace(target_path)
    log_success(f"Metadata applied to {target_path}", indent=indent)
    return 0
