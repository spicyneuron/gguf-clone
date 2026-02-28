from pathlib import Path

from gguf_clone.mlx_params import (
    MlxParams,
    convert_gguf_type,
    load_mlx_params,
    save_mlx_params,
)


# -- convert_gguf_type --


def test_convert_standard_quant_types() -> None:
    assert convert_gguf_type("Q4_K") == 4
    assert convert_gguf_type("Q2_K") == 2
    assert convert_gguf_type("Q8_0") == 8
    assert convert_gguf_type("IQ4_XS") == 4


def test_convert_float_types() -> None:
    assert convert_gguf_type("F16") == "float16"
    assert convert_gguf_type("F32") == "float32"


def test_convert_generic_q_pattern() -> None:
    assert convert_gguf_type("Q5_K_S") == 5
    assert convert_gguf_type("Q3_K_L") == 3


def test_convert_unknown_type_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown GGUF type"):
        _ = convert_gguf_type("NOPE")


# -- save / load round-trip --


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    params = MlxParams(
        default_bits=4,
        tensor_types={
            "lm_head": 6,
            "layers\\.(\\d+)\\.self_attn\\.q_proj": 4,
        },
    )
    path = tmp_path / "params-mlx.json"
    save_mlx_params(params, path)

    loaded = load_mlx_params(path)
    assert loaded is not None
    assert loaded.default_bits == 4
    assert loaded.tensor_types == params.tensor_types


def test_load_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    _ = path.write_text("not json")
    assert load_mlx_params(path) is None


def test_load_missing_default_bits(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    _ = path.write_text('{"tensor_types": {}}')
    assert load_mlx_params(path) is None


def test_load_missing_tensor_types(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    _ = path.write_text('{"default_bits": 4}')
    assert load_mlx_params(path) is None
