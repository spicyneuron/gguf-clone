from gguf_clone.mlx_discover import (
    _classify_gguf_component,  # pyright: ignore[reportPrivateUsage]
    _classify_mlx_component,  # pyright: ignore[reportPrivateUsage]
    build_mapping,
    extract_gguf_components,
    extract_mlx_components,
)


# -- GGUF component extraction --


def test_extract_gguf_components_standard() -> None:
    tensor_types = [
        "output\\.weight=Q6_K",
        "blk\\.(\\d+)\\.attn_q\\.weight=Q4_K",
        "blk\\.(\\d+)\\.ffn_down\\.weight=Q5_K",
    ]
    result = extract_gguf_components(tensor_types)
    assert set(result) == {"attn_q", "ffn_down", "output"}


def test_extract_gguf_components_deduplicates() -> None:
    tensor_types = [
        "blk\\.(0|1)\\.attn_q\\.weight=Q4_K",
        "blk\\.(2|3)\\.attn_q\\.weight=Q2_K",
    ]
    result = extract_gguf_components(tensor_types)
    assert result == ["attn_q"]


# -- MLX component extraction --


def test_extract_mlx_components() -> None:
    paths = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    result = extract_mlx_components(paths)
    assert "self_attn.q_proj" in result
    assert "self_attn.k_proj" in result
    assert "mlp.down_proj" in result
    assert "embed_tokens" in result
    assert "lm_head" in result


# -- Role classification --


def test_classify_gguf_attn() -> None:
    assert _classify_gguf_component("attn_q") == "attn_q"
    assert _classify_gguf_component("attn_k") == "attn_k"
    assert _classify_gguf_component("attn_v") == "attn_v"
    assert _classify_gguf_component("attn_output") == "attn_output"


def test_classify_gguf_ffn() -> None:
    assert _classify_gguf_component("ffn_up") == "mlp_up"
    assert _classify_gguf_component("ffn_down") == "mlp_down"
    assert _classify_gguf_component("ffn_gate") == "mlp_gate"


def test_classify_gguf_moe() -> None:
    assert _classify_gguf_component("ffn_gate_inp") == "moe_switch"
    assert _classify_gguf_component("ffn_up_exps") == "mlp_up"
    assert _classify_gguf_component("ffn_down_exps") == "mlp_down"
    assert _classify_gguf_component("ffn_gate_exps") == "mlp_gate"


def test_classify_gguf_top_level() -> None:
    assert _classify_gguf_component("output") == "lm_head"
    assert _classify_gguf_component("token_embd") == "embed"


def test_classify_mlx_attn() -> None:
    assert _classify_mlx_component("self_attn.q_proj") == "attn_q"
    assert _classify_mlx_component("self_attn.k_proj") == "attn_k"
    assert _classify_mlx_component("self_attn.v_proj") == "attn_v"
    assert _classify_mlx_component("self_attn.o_proj") == "attn_output"


def test_classify_mlx_mlp() -> None:
    assert _classify_mlx_component("mlp.up_proj") == "mlp_up"
    assert _classify_mlx_component("mlp.down_proj") == "mlp_down"
    assert _classify_mlx_component("mlp.gate_proj") == "mlp_gate"


def test_classify_mlx_top_level() -> None:
    assert _classify_mlx_component("lm_head") == "lm_head"
    assert _classify_mlx_component("embed_tokens") == "embed"


# -- build_mapping --


def test_build_mapping_standard_arch() -> None:
    gguf_components = ["attn_q", "attn_k", "attn_v", "attn_output",
                       "ffn_up", "ffn_down", "ffn_gate", "output", "token_embd"]
    mlx_paths = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    result = build_mapping(gguf_components, mlx_paths)

    assert result.component_map == {
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "output": "lm_head",
        "token_embd": "embed_tokens",
    }
    assert result.warnings == []


def test_build_mapping_unmatched_warns() -> None:
    gguf_components = ["attn_q", "unknown_comp"]
    mlx_paths = [
        "model.layers.0.self_attn.q_proj.weight",
    ]
    result = build_mapping(gguf_components, mlx_paths)

    assert "attn_q" in result.component_map
    assert "unknown_comp" not in result.component_map
    assert any("unknown_comp" in w for w in result.warnings)


def test_build_mapping_moe_arch() -> None:
    gguf_components = ["ffn_gate_inp", "ffn_up_exps", "ffn_down_exps", "ffn_gate_exps"]
    mlx_paths = [
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlx.switch_mlp.up_proj.weight",
        "model.layers.0.mlp.switch_mlp.down_proj.weight",
        "model.layers.0.mlp.switch_mlp.gate_proj.weight",
    ]
    result = build_mapping(gguf_components, mlx_paths)

    assert result.component_map["ffn_gate_inp"] == "mlp.gate"
    assert result.component_map["ffn_up_exps"] == "mlx.switch_mlp.up_proj"
    assert result.component_map["ffn_down_exps"] == "mlp.switch_mlp.down_proj"
    assert result.component_map["ffn_gate_exps"] == "mlp.switch_mlp.gate_proj"
