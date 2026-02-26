from pathlib import Path
from typing import cast

import numpy as np
from gguf import GGUFEndian, GGUFReader, GGUFValueType, GGUFWriter, Keys

from gguf_clone.metadata import apply_metadata

MetadataEntry = tuple[object, GGUFValueType, GGUFValueType | None]
MetadataMap = dict[str, MetadataEntry]


def write_gguf(path: Path, *, arch: str, metadata: MetadataMap) -> None:
    writer = GGUFWriter(path, arch=arch, endianess=GGUFEndian.LITTLE)
    for key, (value, value_type, sub_type) in metadata.items():
        writer.add_key_value(key, value, value_type, sub_type=sub_type)

    tensor = np.zeros((1,), dtype=np.float32)
    writer.add_tensor_info(
        "tensor",
        tensor.shape,
        tensor.dtype,
        tensor.nbytes,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.write_tensor_data(tensor)
    writer.close()


def read_metadata(path: Path, key: str) -> object:
    reader = GGUFReader(str(path), "r")
    field = reader.get_field(key)
    assert field is not None
    return cast(object, field.contents())


def test_apply_metadata_add_new_keys(tmp_path: Path) -> None:
    target_path = tmp_path / "target.gguf"
    write_gguf(
        target_path,
        arch="test-arch",
        metadata={
            Keys.General.NAME: ("Test Model", GGUFValueType.STRING, None),
        },
    )

    result = apply_metadata(
        target_path,
        {
            Keys.General.QUANTIZATION_VERSION: "2",
            "general.custom_field": "custom_value",
        },
    )

    assert result == 0
    assert read_metadata(target_path, Keys.General.NAME) == "Test Model"
    assert read_metadata(target_path, Keys.General.QUANTIZATION_VERSION) == "2"
    assert read_metadata(target_path, "general.custom_field") == "custom_value"
    assert read_metadata(target_path, Keys.General.ARCHITECTURE) == "test-arch"


def test_apply_metadata_overwrite_existing(tmp_path: Path) -> None:
    target_path = tmp_path / "target.gguf"
    write_gguf(
        target_path,
        arch="test-arch",
        metadata={
            Keys.General.NAME: ("Old Name", GGUFValueType.STRING, None),
            Keys.General.QUANTIZATION_VERSION: ("1", GGUFValueType.STRING, None),
        },
    )

    result = apply_metadata(
        target_path,
        {
            Keys.General.NAME: "New Name",
            Keys.General.QUANTIZATION_VERSION: "2",
        },
    )

    assert result == 0
    assert read_metadata(target_path, Keys.General.NAME) == "New Name"
    assert read_metadata(target_path, Keys.General.QUANTIZATION_VERSION) == "2"


def test_apply_metadata_empty_dict(tmp_path: Path) -> None:
    target_path = tmp_path / "target.gguf"
    write_gguf(
        target_path,
        arch="test-arch",
        metadata={
            Keys.General.NAME: ("Test Model", GGUFValueType.STRING, None),
        },
    )

    result = apply_metadata(target_path, {})

    assert result == 0
    assert read_metadata(target_path, Keys.General.NAME) == "Test Model"


def test_apply_metadata_preserves_other_keys(tmp_path: Path) -> None:
    target_path = tmp_path / "target.gguf"
    write_gguf(
        target_path,
        arch="test-arch",
        metadata={
            Keys.General.NAME: ("Test Model", GGUFValueType.STRING, None),
            Keys.Tokenizer.CHAT_TEMPLATE: ("{{ chat }}", GGUFValueType.STRING, None),
            "custom.existing": ("existing_value", GGUFValueType.STRING, None),
        },
    )

    result = apply_metadata(
        target_path,
        {
            Keys.General.QUANTIZATION_VERSION: "2",
        },
    )

    assert result == 0
    assert read_metadata(target_path, Keys.General.NAME) == "Test Model"
    assert read_metadata(target_path, Keys.Tokenizer.CHAT_TEMPLATE) == "{{ chat }}"
    assert read_metadata(target_path, "custom.existing") == "existing_value"
    assert read_metadata(target_path, Keys.General.QUANTIZATION_VERSION) == "2"
