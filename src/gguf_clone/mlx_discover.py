"""Discover MLX model parameter paths and map them to GGUF components.

Uses mlx-lm to instantiate the model skeleton (no weights) and
introspect the module tree, replacing hardcoded component maps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING


@dataclass(frozen=True)
class MlxMapping:
    component_map: dict[str, str]  # gguf_component -> mlx_module_path
    warnings: list[str] = field(default_factory=list)


def get_mlx_param_paths(
    model_path: Path, trust_remote_code: bool = False
) -> list[str]:
    """Return sorted unique parameter paths from an MLX model skeleton.

    Instantiates the model from config.json without loading weights.
    """
    try:
        from mlx_lm.utils import load_config as mlx_load_config  # pyright: ignore[reportUnknownVariableType]
    except ImportError as exc:
        msg = "mlx-lm is required for this stage. Install with: uv sync --group mlx"
        raise ImportError(msg) from exc

    from mlx_lm.utils import _get_classes  # pyright: ignore[reportPrivateUsage,reportUnknownVariableType]

    config: dict[str, object] = mlx_load_config(model_path)  # pyright: ignore[reportUnknownVariableType]
    model_class, model_args_class = _get_classes(config=config)  # pyright: ignore[reportAny]

    if trust_remote_code and "model_file" in config:
        msg = "Custom model loading with trust_remote_code is not yet supported."
        raise NotImplementedError(msg)

    model = model_class(model_args_class.from_dict(config))  # pyright: ignore[reportAny]

    if TYPE_CHECKING:
        flat: list[tuple[str, object]] = []
    else:
        from mlx.utils import tree_flatten

        flat = tree_flatten(model.parameters())

    return sorted({path for path, _ in flat})


# -- Role-based matching --

# Keywords that classify a component into a functional role
_ROLE_KEYWORDS: dict[str, list[list[str]]] = {
    "attn_q": [["q"], ["query"]],
    "attn_k": [["k"], ["key"]],
    "attn_v": [["v"], ["value"]],
    "attn_output": [["o"], ["output", "attn"]],
    "mlp_up": [["up"]],
    "mlp_down": [["down"]],
    "mlp_gate": [["gate"], ["gate_proj"]],
    "moe_expert": [["expert"]],
    "moe_shared": [["shared"]],
    "moe_switch": [["switch"]],
    "embed": [["embd"], ["embed"]],
    "lm_head": [["lm_head"]],
}


def _classify_gguf_component(component: str) -> str | None:
    """Map a GGUF component name to a role key."""
    c = component.lower()
    if c == "output":
        return "lm_head"
    if c == "token_embd":
        return "embed"
    if "shexp" in c or "shared_expert" in c:
        if "gate" in c and "inp" not in c:
            return "mlp_gate"
        if "up" in c:
            return "mlp_up"
        if "down" in c:
            return "mlp_down"
        return "moe_shared"
    if "gate_inp" in c or "gate_exps" in c:
        if "inp" in c:
            return "moe_switch"
        return "mlp_gate"
    if "exps" in c:
        if "up" in c:
            return "mlp_up"
        if "down" in c:
            return "mlp_down"
        if "gate" in c:
            return "mlp_gate"
        return "moe_expert"
    if "attn" in c:
        if "output" in c:
            return "attn_output"
        if "_q" in c and "_qkv" not in c:
            return "attn_q"
        if "_k" in c:
            return "attn_k"
        if "_v" in c:
            return "attn_v"
    if "ffn" in c:
        if "up" in c:
            return "mlp_up"
        if "down" in c:
            return "mlp_down"
        if "gate" in c:
            return "mlp_gate"
    return None


def _classify_mlx_component(component: str) -> str | None:
    """Map an MLX component path to a role key."""
    c = component.lower()
    if c == "lm_head":
        return "lm_head"
    if "embed_tokens" in c or "embed" in c:
        return "embed"
    if "shared_expert" in c:
        if "gate_proj" in c:
            return "mlp_gate"
        if "up_proj" in c:
            return "mlp_up"
        if "down_proj" in c:
            return "mlp_down"
        return "moe_shared"
    if "switch_mlp" in c or "gate" == c.split(".")[-1]:
        if "gate_proj" in c:
            return "mlp_gate"
        if "up_proj" in c:
            return "mlp_up"
        if "down_proj" in c:
            return "mlp_down"
        return "moe_switch"
    if "self_attn" in c or "attention" in c:
        if "o_proj" in c:
            return "attn_output"
        if "q_proj" in c:
            return "attn_q"
        if "k_proj" in c:
            return "attn_k"
        if "v_proj" in c:
            return "attn_v"
    if "mlp" in c:
        if "up_proj" in c:
            return "mlp_up"
        if "down_proj" in c:
            return "mlp_down"
        if "gate_proj" in c:
            return "mlp_gate"
    return None


def extract_gguf_components(tensor_types: list[str]) -> list[str]:
    """Extract unique GGUF component names from tensor_types entries.

    Input entries look like: blk\\.(\\d+)\\.attn_q\\.weight=Q4_K
    or: output\\.weight=Q6_K
    """
    components: list[str] = []
    seen: set[str] = set()
    for entry in tensor_types:
        if "=" not in entry:
            continue
        pattern = entry.rsplit("=", 1)[0].strip()
        # Unescape regex dots
        clean = pattern.replace("\\.", ".")
        # Check for layered pattern: blk.(digits).component.weight
        m = re.match(r"^blk\.(?:\(.*?\)|\d+)\.(.+?)\.weight$", clean)
        if m:
            comp = m.group(1)
            if comp not in seen:
                components.append(comp)
                seen.add(comp)
            continue
        # Top-level: output.weight, token_embd.weight
        m = re.match(r"^([a-z_]+)\.weight$", clean)
        if m:
            comp = m.group(1)
            if comp not in seen:
                components.append(comp)
                seen.add(comp)
    return components


def extract_mlx_components(mlx_paths: list[str]) -> dict[str, str]:
    """Extract unique MLX component templates from param paths.

    Returns {component_template: full_prefix} where component_template
    is the part after the layer index stripped of .weight/.bias.
    """
    components: dict[str, str] = {}
    for path in mlx_paths:
        # Strip .weight/.bias suffix
        stem = re.sub(r"\.(weight|bias|scales|biases)$", "", path)
        # Layered: model.layers.0.self_attn.q_proj -> self_attn.q_proj
        m = re.match(r"^(?:model\.)?layers\.\d+\.(.+)$", stem)
        if m:
            comp = m.group(1)
            if comp not in components:
                components[comp] = comp
            continue
        # Top-level components
        # model.embed_tokens -> embed_tokens, lm_head -> lm_head
        top = re.sub(r"^model\.", "", stem)
        if top not in components:
            components[top] = top
    return components


def build_mapping(
    gguf_components: list[str], mlx_paths: list[str]
) -> MlxMapping:
    """Match GGUF tensor component names to MLX module paths by role."""
    mlx_components = extract_mlx_components(mlx_paths)
    warnings: list[str] = []
    component_map: dict[str, str] = {}

    # Build role -> mlx_component lookup
    role_to_mlx: dict[str, list[str]] = {}
    for mlx_comp in mlx_components:
        role = _classify_mlx_component(mlx_comp)
        if role:
            role_to_mlx.setdefault(role, []).append(mlx_comp)

    for gguf_comp in gguf_components:
        role = _classify_gguf_component(gguf_comp)
        if role is None:
            warnings.append(f"No role inferred for GGUF component: {gguf_comp}")
            continue
        candidates = role_to_mlx.get(role, [])
        if not candidates:
            warnings.append(
                f"No MLX match for GGUF component '{gguf_comp}' (role: {role})"
            )
            continue
        if len(candidates) > 1:
            warnings.append(
                f"Multiple MLX matches for '{gguf_comp}' (role: {role}): {candidates}; using first"
            )
        component_map[gguf_comp] = candidates[0]

    return MlxMapping(component_map=component_map, warnings=warnings)
