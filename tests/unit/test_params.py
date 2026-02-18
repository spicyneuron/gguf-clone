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


def _make_tensor(name: str, qtype_code: int = 2) -> MagicMock:
    t = MagicMock()
    t.name = name
    t.tensor_type = qtype_code
    t.shape = [10, 10]
    return t


def _qtype_factory(mapping: dict[int, str]) -> MagicMock:
    """Return a side_effect callable that maps int codes to named mocks."""

    def factory(code: int) -> MagicMock:
        m = MagicMock()
        m.name = mapping.get(code, "UNKNOWN")
        return m

    return cast(MagicMock, factory)


def _matches(tensor_type: str, name: str, qtype: str) -> bool:
    if not tensor_type.endswith(f"={qtype}"):
        return False
    pattern = tensor_type[: -(len(qtype) + 1)]
    return re.fullmatch(pattern, name) is not None


def test_build_params_grouping() -> None:
    """Mixed qtypes across layers produce literal per-layer entries."""
    with patch("gguf_clone.params.GGUFReader") as MockReader:
        mock_instance = cast(MagicMock, MockReader.return_value)
        with patch("gguf_clone.params.GGMLQuantizationType") as MockQType:
            MockQType.side_effect = _qtype_factory({2: "Q4_K", 3: "Q6_K"})

            mock_instance.tensors = [
                _make_tensor("blk.0.attn_q.weight", 2),
                _make_tensor("blk.1.attn_q.weight", 2),
                _make_tensor("blk.2.attn_q.weight", 3),
                _make_tensor("output.weight", 2),
                _make_tensor("token_embd.weight", 2),
            ]

            params = build_params(Path("dummy.gguf"))
            types = params.tensor_types

            # Mixed layers -> grouped regex by qtype
            assert any(_matches(t, "blk.0.attn_q.weight", "Q4_K") for t in types)
            assert any(_matches(t, "blk.1.attn_q.weight", "Q4_K") for t in types)
            assert any(_matches(t, "blk.2.attn_q.weight", "Q6_K") for t in types)
            assert "output.weight=Q4_K" in types
            assert "token_embd.weight=Q4_K" in types
            assert params.default_type == "Q4_K"


def test_build_params_ignore() -> None:
    """Structural filters still apply; IGNORE list no longer exists."""
    with patch("gguf_clone.params.GGUFReader") as MockReader:
        mock_instance = cast(MagicMock, MockReader.return_value)
        with patch("gguf_clone.params.GGMLQuantizationType") as MockQType:
            MockQType.side_effect = _qtype_factory({1: "Q4_K"})

            t1 = MagicMock()
            t1.name = "blk.0.ffn_gate_inp.weight"  # No longer ignored
            t1.tensor_type = 1
            t1.shape = [10, 10]

            t2 = MagicMock()
            t2.name = "valid.weight"
            t2.tensor_type = 1
            t2.shape = [10, 10]

            t3 = MagicMock()
            t3.name = "bias.tensor"  # Not ending in weight -> ignored
            t3.tensor_type = 1
            t3.shape = [10, 10]

            mock_instance.tensors = [t1, t2, t3]

            params = build_params(Path("dummy.gguf"))

            # ffn_gate_inp.weight is now kept (layered), valid.weight kept
            assert len(params.tensor_types) == 2
            assert any(
                _matches(t, "blk.0.ffn_gate_inp.weight", "Q4_K")
                for t in params.tensor_types
            )
            assert "valid.weight=Q4_K" in params.tensor_types


def test_build_params_multiple_ggufs() -> None:
    """Same qtype across files -> single wildcard entry."""
    with patch("gguf_clone.params.GGUFReader") as MockReader:
        reader_one = MagicMock()
        reader_two = MagicMock()
        MockReader.side_effect = [reader_one, reader_two]
        with patch("gguf_clone.params.GGMLQuantizationType") as MockQType:
            MockQType.side_effect = _qtype_factory({2: "Q4_K"})

            reader_one.tensors = [_make_tensor("blk.0.attn_q.weight", 2)]
            reader_two.tensors = [_make_tensor("blk.1.attn_q.weight", 2)]

            params = build_params([Path("a.gguf"), Path("b.gguf")])
            types = params.tensor_types

            # Both layers same type -> single wildcard
            assert len(types) == 1
            assert _matches(types[0], "blk.0.attn_q.weight", "Q4_K")
            assert _matches(types[0], "blk.1.attn_q.weight", "Q4_K")
            assert _matches(types[0], "blk.99.attn_q.weight", "Q4_K")
            assert params.default_type == "Q4_K"


def test_build_params_conflict_raises() -> None:
    """Conflicting qtypes for the same layer across files raises ValueError."""
    with patch("gguf_clone.params.GGUFReader") as MockReader:
        reader_one = MagicMock()
        reader_two = MagicMock()
        MockReader.side_effect = [reader_one, reader_two]
        with patch("gguf_clone.params.GGMLQuantizationType") as MockQType:
            MockQType.side_effect = _qtype_factory({2: "Q4_K", 3: "Q6_K"})

            # Same tensor, different qtypes across files
            reader_one.tensors = [_make_tensor("blk.0.attn_q.weight", 2)]
            reader_two.tensors = [_make_tensor("blk.0.attn_q.weight", 3)]

            with pytest.raises(ValueError, match="Conflicting types"):
                build_params([Path("a.gguf"), Path("b.gguf")])  # pyright: ignore[reportUnusedCallResult]


def test_build_params_uniform_wildcard() -> None:
    """All layers sharing a qtype produce a single \\d+ wildcard."""
    with patch("gguf_clone.params.GGUFReader") as MockReader:
        mock_instance = cast(MagicMock, MockReader.return_value)
        with patch("gguf_clone.params.GGMLQuantizationType") as MockQType:
            MockQType.side_effect = _qtype_factory({2: "Q4_K"})

            mock_instance.tensors = [
                _make_tensor("blk.0.ffn_up.weight", 2),
                _make_tensor("blk.1.ffn_up.weight", 2),
                _make_tensor("blk.2.ffn_up.weight", 2),
            ]

            params = build_params(Path("dummy.gguf"))
            types = params.tensor_types

            assert len(types) == 1
            assert r"(\d+)" in types[0]
            assert _matches(types[0], "blk.0.ffn_up.weight", "Q4_K")
            assert _matches(types[0], "blk.42.ffn_up.weight", "Q4_K")


def test_build_params_mixed_grouped() -> None:
    """Layers with different qtypes produce grouped regex by qtype."""
    with patch("gguf_clone.params.GGUFReader") as MockReader:
        mock_instance = cast(MagicMock, MockReader.return_value)
        with patch("gguf_clone.params.GGMLQuantizationType") as MockQType:
            MockQType.side_effect = _qtype_factory({2: "Q4_K", 3: "Q6_K"})

            mock_instance.tensors = [
                _make_tensor("blk.0.ffn_up.weight", 2),
                _make_tensor("blk.1.ffn_up.weight", 3),
            ]

            params = build_params(Path("dummy.gguf"))
            types = params.tensor_types

            assert len(types) == 2
            assert _matches(types[0], "blk.0.ffn_up.weight", "Q4_K")
            assert _matches(types[1], "blk.1.ffn_up.weight", "Q6_K")
            # No wildcard -- grouped alternation instead
            assert not any(r"(\d+)" in t for t in types)


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
