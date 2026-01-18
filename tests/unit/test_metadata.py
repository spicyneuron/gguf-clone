from pathlib import Path
from typing import cast

import numpy as np
from gguf import GGUFEndian, GGUFReader, GGUFValueType, GGUFWriter, Keys

from gguf_clone.metadata import copy_template_metadata

MetadataEntry = tuple[object, GGUFValueType, GGUFValueType | None]
MetadataMap = dict[str, MetadataEntry]


def write_gguf(path: Path, *, arch: str, metadata: MetadataMap) -> None:
    writer = GGUFWriter(path, arch=arch, endianess=GGUFEndian.LITTLE)
    for key, (value, value_type, sub_type) in metadata.items():
        writer.add_key_value(key, value, value_type, sub_type=sub_type)

    tensor = np.zeros((1,), dtype=np.float32)
    writer.add_tensor_info(  # pyright: ignore[reportUnknownMemberType]
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


def test_copy_template_metadata_chat_template(tmp_path: Path) -> None:
    template_path = tmp_path / "template.gguf"
    target_path = tmp_path / "target.gguf"
    write_gguf(
        template_path,
        arch="template-arch",
        metadata={
            Keys.Tokenizer.CHAT_TEMPLATE: (
                "{{ template }}",
                GGUFValueType.STRING,
                None,
            ),
        },
    )
    write_gguf(
        target_path,
        arch="target-arch",
        metadata={
            Keys.Tokenizer.CHAT_TEMPLATE: ("old-template", GGUFValueType.STRING, None),
            Keys.General.NAME: ("Target Model", GGUFValueType.STRING, None),
        },
    )

    result = copy_template_metadata(
        template_path,
        target_path,
        [Keys.Tokenizer.CHAT_TEMPLATE],
    )

    assert result == 0
    assert read_metadata(target_path, Keys.Tokenizer.CHAT_TEMPLATE) == "{{ template }}"
    assert read_metadata(target_path, Keys.General.NAME) == "Target Model"
    assert read_metadata(target_path, Keys.General.ARCHITECTURE) == "target-arch"
