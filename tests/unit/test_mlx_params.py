from pathlib import Path

from gguf_clone.mlx_params import (
    MlxParams,
    build_component_map,
    convert_gguf_params,
    convert_gguf_type,
    convert_pattern,
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


# -- convert_pattern --


def test_convert_top_level_output() -> None:
    cmap = build_component_map()
    assert convert_pattern("output\\.weight", cmap) == "lm_head"
    assert convert_pattern("token_embd\\.weight", cmap) == "embed_tokens"


def test_convert_layer_pattern() -> None:
    cmap = build_component_map()
    result = convert_pattern("blk\\.(\\d+)\\.attn_q\\.weight", cmap)
    assert result == "layers\\.(\\d+)\\.self_attn\\.q_proj"


def test_convert_mlp_pattern() -> None:
    cmap = build_component_map()
    result = convert_pattern("blk\\.(\\d+)\\.ffn_down\\.weight", cmap)
    assert result == "layers\\.(\\d+)\\.mlp\\.down_proj"


def test_convert_moe_pattern() -> None:
    cmap = build_component_map()
    result = convert_pattern("blk\\.(\\d+)\\.ffn_gate_exps\\.weight", cmap)
    assert result == "layers\\.(\\d+)\\.mlp\\.switch_mlp\\.gate_proj"


def test_convert_unknown_component_returns_none() -> None:
    cmap = build_component_map()
    assert convert_pattern("blk\\.(\\d+)\\.unknown_thing\\.weight", cmap) is None


def test_convert_non_blk_unknown_returns_none() -> None:
    cmap = build_component_map()
    assert convert_pattern("some_random_tensor\\.weight", cmap) is None


def test_convert_arch_override() -> None:
    cmap = build_component_map("deepseek_v3")
    result = convert_pattern("blk\\.(\\d+)\\.attn_q_a\\.weight", cmap)
    assert result == "layers\\.(\\d+)\\.self_attn\\.q_a_proj"


def test_convert_arch_longest_match() -> None:
    """attn_q should not partially match attn_qkv."""
    cmap = build_component_map("qwen3_next")
    result = convert_pattern("blk\\.(\\d+)\\.attn_qkv\\.weight", cmap)
    assert result == "layers\\.(\\d+)\\.linear_attn\\.in_proj_qkvz"


# -- convert_gguf_params --


def test_convert_full_recipe() -> None:
    tensor_types = [
        "output\\.weight=Q6_K",
        "blk\\.(\\d+)\\.attn_q\\.weight=Q4_K",
        "blk\\.(\\d+)\\.ffn_down\\.weight=Q5_K",
    ]
    result = convert_gguf_params(tensor_types, "Q4_K")

    assert result.default_bits == 4
    assert result.tensor_types == {
        "lm_head": 6,
        "layers\\.(\\d+)\\.self_attn\\.q_proj": 4,
        "layers\\.(\\d+)\\.mlp\\.down_proj": 5,
    }
    assert result.warnings == []


def test_convert_recipe_with_unmapped_warns() -> None:
    tensor_types = [
        "blk\\.(\\d+)\\.attn_q\\.weight=Q4_K",
        "blk\\.(\\d+)\\.unknown_comp\\.weight=Q4_K",
    ]
    result = convert_gguf_params(tensor_types, "Q4_K")

    assert len(result.tensor_types) == 1
    assert len(result.warnings) == 1
    assert "Unmapped" in result.warnings[0]


def test_convert_recipe_unknown_default_type() -> None:
    result = convert_gguf_params([], "WEIRD_TYPE")
    assert result.default_bits == 4
    assert any("Unknown default_type" in w for w in result.warnings)


def test_convert_recipe_float_default_type() -> None:
    result = convert_gguf_params([], "F16")
    assert result.default_bits == 4
    assert any("float dtype" in w for w in result.warnings)


def test_convert_recipe_malformed_entry() -> None:
    tensor_types = ["no_equals_sign"]
    result = convert_gguf_params(tensor_types, "Q4_K")
    assert any("malformed" in w for w in result.warnings)


def test_convert_recipe_with_arch() -> None:
    tensor_types = [
        "blk\\.(\\d+)\\.attn_q_a\\.weight=Q4_K",
    ]
    result = convert_gguf_params(tensor_types, "Q4_K", arch="deepseek_v3")
    assert "layers\\.(\\d+)\\.self_attn\\.q_a_proj" in result.tensor_types


def test_convert_recipe_with_layer_groups() -> None:
    """Alternation groups (0|1|2) should convert correctly."""
    tensor_types = [
        "blk\\.(0|1|2)\\.attn_q\\.weight=Q4_K",
        "blk\\.(3|4|5)\\.attn_q\\.weight=Q2_K",
    ]
    result = convert_gguf_params(tensor_types, "Q4_K")
    assert result.tensor_types == {
        "layers\\.(0|1|2)\\.self_attn\\.q_proj": 4,
        "layers\\.(3|4|5)\\.self_attn\\.q_proj": 2,
    }


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
