import re
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from gguf_clone.main import quant_label_from_stem
from gguf_clone.params import build_params, copy_imatrix


@pytest.mark.parametrize(
    "stem,expected",
    [
        ("model-Q4_K_M", "Q4_K_M"),
        ("model-UD-IQ2_XXS", "UD-IQ2_XXS"),
        ("model-F16", "F16"),
        ("model-unknown", None),
        ("Mixed-Case-q4_k_m", "Q4_K_M"),
    ],
)
def test_quant_label_from_stem(stem: str, expected: str | None) -> None:
    assert quant_label_from_stem(stem) == expected


def test_build_params_grouping() -> None:
    # Mock ReaderTensor
    def make_tensor(name: str, qtype_code: int = 2) -> MagicMock:
        # 2 is usually Q4_0 or similar in gguf enum, treating as generic
        t = MagicMock()
        t.name = name
        t.tensor_type = qtype_code
        t.shape = [10, 10]  # valid shape > 1D
        return t

    # Mock GGUFReader
    with patch("gguf_clone.params.GGUFReader") as MockReader:
        mock_reader = cast(MagicMock, MockReader)
        mock_instance = cast(MagicMock, mock_reader.return_value)
        # Mock GGMLQuantizationType.name lookup
        with patch("gguf_clone.params.GGMLQuantizationType") as MockQType:
            mock_qtype = cast(MagicMock, MockQType)

            # Map integer type to string name
            def make_qtype(name: str) -> MagicMock:
                m = MagicMock()
                m.name = name
                return m

            def qtype_for_code(code: int) -> MagicMock:
                return make_qtype("Q4_K") if code == 2 else make_qtype("Q6_K")

            mock_qtype.side_effect = qtype_for_code

            mock_instance.tensors = [
                make_tensor("blk.0.attn_q.weight", 2),
                make_tensor("blk.1.attn_q.weight", 2),
                make_tensor("blk.2.attn_q.weight", 3),  # Different type
                make_tensor("output.weight", 2),
                make_tensor(
                    "token_embd.weight", 2
                ),  # Should be ignored by ignore_tensor defaults? No, ignore_tensor logic:
                # ignore_tensor: ends with weight? yes. shape < 2? no. in IGNORE?
                # IGNORE has "time_mix_...", "attn_rel_b.weight". Not "token_embd.weight".
                # Wait, let's check IGNORE list in params.py.
            ]

            # Re-read IGNORE list logic in params.py to be sure about token_embd.weight
            # IGNORE = [..., ".position_embd.", ...]
            # token_embd is usually not ignored unless explicitly listed.

            params = build_params(Path("dummy.gguf"))

            # Check grouping
            # blk.0.attn_q.weight=Q4_K
            # blk.1.attn_q.weight=Q4_K
            # blk.2.attn_q.weight=Q6_K
            # Expected regex: blk\.(0|1)\.attn_q\.weight=Q4_K
            # And: blk\.2\.attn_q\.weight=Q6_K
            # output.weight=Q4_K (no group)

            def matches(tensor_type: str, name: str, qtype: str) -> bool:
                if not tensor_type.endswith(f"={qtype}"):
                    return False
                pattern = tensor_type[: -(len(qtype) + 1)]
                return re.fullmatch(pattern, name) is not None

            types = params.tensor_types
            assert any(matches(t, "blk.0.attn_q.weight", "Q4_K") for t in types)
            assert any(matches(t, "blk.1.attn_q.weight", "Q4_K") for t in types)
            assert any(matches(t, "blk.2.attn_q.weight", "Q6_K") for t in types)
            assert "output.weight=Q4_K" in types

            # Default type should be Q4_K (majority)
            assert params.default_type == "Q4_K"


def test_build_params_ignore() -> None:
    with patch("gguf_clone.params.GGUFReader") as MockReader:
        mock_reader = cast(MagicMock, MockReader)
        mock_instance = cast(MagicMock, mock_reader.return_value)
        with patch("gguf_clone.params.GGMLQuantizationType") as MockQType:
            mock_qtype = cast(MagicMock, MockQType)

            def make_qtype(name: str) -> MagicMock:
                m = MagicMock()
                m.name = name
                return m

            def qtype_for_code(_code: int) -> MagicMock:
                return make_qtype("Q4_K")

            mock_qtype.side_effect = qtype_for_code

            t1 = MagicMock()
            t1.name = "blk.0.ffn_gate_inp.weight"  # In IGNORE list
            t1.tensor_type = 1
            t1.shape = [10, 10]

            t2 = MagicMock()
            t2.name = "valid.weight"
            t2.tensor_type = 1
            t2.shape = [10, 10]

            t3 = MagicMock()
            t3.name = "bias.tensor"  # Not ending in weight
            t3.tensor_type = 1
            t3.shape = [10, 10]

            mock_instance.tensors = [t1, t2, t3]

            params = build_params(Path("dummy.gguf"))

            # Only valid.weight should remain
            assert len(params.tensor_types) == 1
            assert params.tensor_types[0] == "valid.weight=Q4_K"


def test_build_params_multiple_ggufs() -> None:
    def make_tensor(name: str, qtype_code: int = 2) -> MagicMock:
        t = MagicMock()
        t.name = name
        t.tensor_type = qtype_code
        t.shape = [10, 10]
        return t

    with patch("gguf_clone.params.GGUFReader") as MockReader:
        reader_one = MagicMock()
        reader_two = MagicMock()
        MockReader.side_effect = [reader_one, reader_two]
        with patch("gguf_clone.params.GGMLQuantizationType") as MockQType:
            mock_qtype = cast(MagicMock, MockQType)

            def make_qtype(name: str) -> MagicMock:
                m = MagicMock()
                m.name = name
                return m

            def qtype_for_code(_code: int) -> MagicMock:
                return make_qtype("Q4_K")

            mock_qtype.side_effect = qtype_for_code

            reader_one.tensors = [make_tensor("blk.0.attn_q.weight", 2)]
            reader_two.tensors = [make_tensor("blk.1.attn_q.weight", 2)]

            params = build_params([Path("a.gguf"), Path("b.gguf")])

            def matches(tensor_type: str, name: str, qtype: str) -> bool:
                if not tensor_type.endswith(f"={qtype}"):
                    return False
                pattern = tensor_type[: -(len(qtype) + 1)]
                return re.fullmatch(pattern, name) is not None

            types = params.tensor_types
            assert any(matches(t, "blk.0.attn_q.weight", "Q4_K") for t in types)
            assert any(matches(t, "blk.1.attn_q.weight", "Q4_K") for t in types)
            assert params.default_type == "Q4_K"


def test_copy_imatrix_copies_file(tmp_path: Path) -> None:
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    src_file = src_dir / "imatrix.dat"
    _ = src_file.write_bytes(b"imatrix data")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    result = copy_imatrix(src_file, dest_dir)

    assert result == "imatrix.dat"
    assert (dest_dir / "imatrix.dat").exists()
    assert (dest_dir / "imatrix.dat").read_bytes() == b"imatrix data"


def test_copy_imatrix_with_prefix(tmp_path: Path) -> None:
    src_file = tmp_path / "deep" / "nested" / "path" / "my-imatrix.dat"
    src_file.parent.mkdir(parents=True)
    _ = src_file.write_bytes(b"data")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    result = copy_imatrix(src_file, dest_dir, prefix="params")

    assert result == "params/my-imatrix.dat"
    assert (dest_dir / "my-imatrix.dat").exists()


def test_copy_imatrix_skips_if_same_file(tmp_path: Path) -> None:
    imatrix = tmp_path / "imatrix.dat"
    _ = imatrix.write_bytes(b"original")

    result = copy_imatrix(imatrix, tmp_path)

    assert result == "imatrix.dat"
    assert imatrix.read_bytes() == b"original"


def test_copy_imatrix_overwrites_different_file(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src_file = src_dir / "imatrix.dat"
    _ = src_file.write_bytes(b"new content")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    dest_file = dest_dir / "imatrix.dat"
    _ = dest_file.write_bytes(b"old content")

    result = copy_imatrix(src_file, dest_dir, prefix="params")

    assert result == "params/imatrix.dat"
    assert dest_file.read_bytes() == b"new content"
