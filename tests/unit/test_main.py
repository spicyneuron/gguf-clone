from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from gguf_clone.main import RunConfig, run
from gguf_clone.params import ParamsPayload, QuantParams
from gguf_clone.resolve import ResolvedModels, ToolPaths


def _make_config(*, split: str = "50G") -> RunConfig:
    return RunConfig(
        template_repo="org/template",
        template_gguf_patterns=["*.gguf"],
        template_imatrix_pattern="*imatrix*",
        template_copy_metadata=[],
        template_copy_files=[],
        target_repo="org/target",
        target_exclude_files=[],
        output_prefix="",
        output_split=split,
        output_converted_dir="converted",
        output_params_dir="params",
        output_quantized_dir="quantized",
        output_apply_metadata={},
    )


def _make_resolved(tmp_path: Path) -> ResolvedModels:
    template_snapshot = tmp_path / "template"
    template_snapshot.mkdir()
    template_gguf = template_snapshot / "template-Q4_K.gguf"
    _ = template_gguf.write_bytes(b"template")
    template_imatrix = template_snapshot / "imatrix.dat"
    _ = template_imatrix.write_bytes(b"imatrix")
    target_snapshot = tmp_path / "target"
    target_snapshot.mkdir()
    return ResolvedModels(
        template_snapshot=template_snapshot,
        template_imatrix=template_imatrix,
        template_ggufs=[[template_gguf]],
        target_snapshot=target_snapshot,
    )


def _make_tools() -> ToolPaths:
    return ToolPaths(
        llama_quantize=Path("/bin/llama-quantize"),
        llama_gguf_split=Path("/bin/llama-gguf-split"),
        convert_hf_to_gguf=Path("/bin/convert_hf_to_gguf.py"),
    )


def _fake_convert(
    _model_path: Path,
    *,
    output_dir: Path,
    outfile_name: str | None,
    **_kwargs: object,
) -> int:
    assert outfile_name is not None
    _ = (output_dir / outfile_name).write_bytes(b"converted")
    return 0


def _fake_quantize(
    _input_path: Path,
    output_path: Path,
    *,
    tensor_types: list[str],
    default_type: str,
    imatrix: str,
    llama_quantize: Path,
    cwd: Path | None = None,
    indent: str = "",
) -> int:
    del tensor_types, default_type, imatrix, llama_quantize, cwd, indent
    _ = output_path.write_bytes(b"q")
    return 0


def _patch_base(config: RunConfig, resolved: ResolvedModels, tools: ToolPaths) -> list[object]:
    return [
        patch("gguf_clone.main.check_deps", return_value=[]),
        patch("gguf_clone.main.check_gguf_support", return_value=None),
        patch("gguf_clone.main.load_config", return_value=config),
        patch("gguf_clone.main.resolve_tools", return_value=tools),
        patch("gguf_clone.main.resolve_models", return_value=resolved),
        patch("gguf_clone.main.convert_target", side_effect=_fake_convert),
    ]


def test_run_use_existing_params_skips_preconfirm_work(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    _ = config_path.write_text("stub")
    config = _make_config()
    resolved = _make_resolved(tmp_path)
    tools = _make_tools()
    params_path = tmp_path / "params" / "org-template-Q4_K.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    _ = params_path.write_text("{}")
    payload = ParamsPayload(
        tensor_types=["tensor.weight=Q4_K"],
        default_type="Q4_K",
        imatrix="params/imatrix.dat",
    )

    patches = _patch_base(config, resolved, tools)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patch("gguf_clone.main.load_params", return_value=payload),
        patch("gguf_clone.main.quantize_gguf", side_effect=_fake_quantize),
        patch("gguf_clone.main.copy_imatrix") as copy_mock,
        patch("gguf_clone.main.build_params_payload") as build_mock,
    ):
        result = run(config_path, overwrite_behavior="use")

    assert result == 0
    copy_mock.assert_not_called()
    build_mock.assert_not_called()


def test_run_overwrite_params_confirms_before_copy_and_build(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    _ = config_path.write_text("stub")
    config = _make_config()
    resolved = _make_resolved(tmp_path)
    tools = _make_tools()
    params_path = tmp_path / "params" / "org-template-Q4_K.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    _ = params_path.write_text("{}")

    events: list[str] = []

    def fake_confirm(paths: list[Path], label: str, *, indent: str = "") -> str:
        del paths, indent
        events.append(f"confirm:{label}")
        return "overwrite"

    def fake_copy(src: Path, dest_dir: Path, prefix: str = "") -> str:
        del src, dest_dir, prefix
        events.append("copy")
        return "params/imatrix.dat"

    def fake_build(paths: Path | list[Path], imatrix: str) -> tuple[ParamsPayload, QuantParams]:
        del paths
        events.append("build")
        payload = ParamsPayload(
            tensor_types=["tensor.weight=Q4_K"],
            default_type="Q4_K",
            imatrix=imatrix,
        )
        params = QuantParams(
            tensor_types=payload.tensor_types,
            default_type=payload.default_type,
            quant_type_counts={"Q4_K": 1},
        )
        return payload, params

    def fake_save(payload: ParamsPayload, output_path: Path) -> None:
        del payload
        events.append("save")
        _ = output_path.write_text("{}")

    def fake_quantize(
        _input_path: Path,
        output_path: Path,
        *,
        tensor_types: list[str],
        default_type: str,
        imatrix: str,
        llama_quantize: Path,
        cwd: Path | None = None,
        indent: str = "",
    ) -> int:
        del tensor_types, default_type, imatrix, llama_quantize, cwd, indent
        events.append("quantize")
        _ = output_path.write_bytes(b"q")
        return 0

    patches = _patch_base(config, resolved, tools)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patch("gguf_clone.main.confirm_overwrite", side_effect=fake_confirm),
        patch("gguf_clone.main.copy_imatrix", side_effect=fake_copy),
        patch("gguf_clone.main.build_params_payload", side_effect=fake_build),
        patch("gguf_clone.main.save_params_payload", side_effect=fake_save),
        patch("gguf_clone.main.quantize_gguf", side_effect=fake_quantize),
    ):
        result = run(config_path)

    assert result == 0
    assert events.index("confirm:params") < events.index("copy")
    assert events.index("copy") < events.index("build")
    assert events.index("build") < events.index("save")


def test_run_removes_existing_splits_before_quantize(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    _ = config_path.write_text("stub")
    config = _make_config()
    resolved = _make_resolved(tmp_path)
    tools = _make_tools()

    split_dir = tmp_path / "quantized" / "Q4_K"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_file = split_dir / "org-target-Q4_K-00001-of-00002.gguf"
    _ = split_file.write_bytes(b"old-split")

    payload = ParamsPayload(
        tensor_types=["tensor.weight=Q4_K"],
        default_type="Q4_K",
        imatrix="params/imatrix.dat",
    )

    events: list[str] = []
    removed: list[list[Path]] = []

    def fake_confirm(paths: list[Path], label: str, *, indent: str = "") -> str:
        del paths, indent
        events.append(f"confirm:{label}")
        if label == "params":
            return "use"
        return "overwrite"

    def fake_remove(paths: list[Path]) -> bool:
        removed.append(list(paths))
        events.append("remove")
        for path in paths:
            if path.exists():
                path.unlink()
        return True

    def fake_quantize_fail(
        _input_path: Path,
        output_path: Path,
        *,
        tensor_types: list[str],
        default_type: str,
        imatrix: str,
        llama_quantize: Path,
        cwd: Path | None = None,
        indent: str = "",
    ) -> int:
        del output_path, tensor_types, default_type, imatrix, llama_quantize, cwd, indent
        events.append("quantize")
        return 1

    patches = _patch_base(config, resolved, tools)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patch("gguf_clone.main.confirm_overwrite", side_effect=fake_confirm),
        patch("gguf_clone.main.load_params", return_value=payload),
        patch("gguf_clone.main.remove_files", side_effect=fake_remove),
        patch("gguf_clone.main.quantize_gguf", side_effect=fake_quantize_fail),
    ):
        result = run(config_path)

    assert result == 1
    assert not split_file.exists()
    assert any(split_file in batch for batch in removed)
    assert events.index("remove") < events.index("quantize")
