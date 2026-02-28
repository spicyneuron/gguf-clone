from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from gguf_clone.artifacts import Artifacts
from gguf_clone.config import RunConfig as V2RunConfig, SourceRef
from gguf_clone.main import run_extract_template, run_pipeline, run_quantize_gguf
from gguf_clone.params import ParamsPayload, QuantParams, load_params
from gguf_clone.resolve import ToolPaths


def _v2_config(
    tmp_path: Path,
    *,
    extract_template: object = None,
    quantize_gguf: object = None,
) -> V2RunConfig:
    from gguf_clone.config import ExtractTemplateConfig, QuantizeGgufConfig

    ep = None
    if extract_template is not None:
        ep = ExtractTemplateConfig.model_validate(extract_template)
    qg = None
    if quantize_gguf is not None:
        qg = QuantizeGgufConfig.model_validate(quantize_gguf)
    return V2RunConfig(
        template=SourceRef(repo="org/template", path=None),
        target=SourceRef(repo="org/target", path=None),
        output_dir=tmp_path / "output",
        extract_template=ep,
        quantize_gguf=qg,
        quantize_mlx=None,
    )


def _v2_artifacts(tmp_path: Path) -> Artifacts:
    return Artifacts(
        output_dir=tmp_path / "output",
        template_slug="org-template",
        target_slug="org-target",
    )


def _tool_paths() -> ToolPaths:
    return ToolPaths(
        llama_quantize=Path("/bin/llama-quantize"),
        llama_gguf_split=Path("/bin/llama-gguf-split"),
        convert_hf_to_gguf=Path("/bin/convert.py"),
    )


def test_run_extract_template_writes_gguf_json(tmp_path: Path) -> None:
    config = _v2_config(
        tmp_path,
        extract_template={"ggufs": ["*Q4_K*.gguf"], "targets": ["gguf"]},
    )
    arts = _v2_artifacts(tmp_path)

    template_snapshot = tmp_path / "template_snap"
    template_snapshot.mkdir()
    _ = (template_snapshot / "model-Q4_K.gguf").write_bytes(b"gguf")

    fake_params = QuantParams(
        tensor_types=["blk\\.(\\d+)\\.attn_q\\.weight=Q4_K"],
        default_type="Q4_K",
        quant_type_counts={"Q4_K": 10},
    )

    with (
        patch("gguf_clone.main.check_deps", return_value=[]),
        patch("gguf_clone.main.check_gguf_support", return_value=None),
        patch("gguf_clone.stages.extract_template.resolve_source_snapshot", return_value=template_snapshot),
        patch("gguf_clone.stages.extract_template.build_params", return_value=fake_params),
    ):
        result = run_extract_template(config, arts, overwrite_behavior="overwrite")

    assert result == 0
    assert arts.params_gguf("Q4_K").exists()


def test_run_extract_template_stages_template_artifacts(tmp_path: Path) -> None:
    config = _v2_config(
        tmp_path,
        extract_template={"ggufs": ["*Q4_K*.gguf"], "targets": ["gguf"]},
        quantize_gguf={
            "imatrix": "*imatrix*",
            "copy_metadata": ["tokenizer.chat_template"],
            "copy_files": ["*mmproj*"],
            "apply_metadata": {},
        },
    )
    arts = _v2_artifacts(tmp_path)

    template_snapshot = tmp_path / "template_snap"
    template_snapshot.mkdir()
    _ = (template_snapshot / "model-Q4_K.gguf").write_bytes(b"gguf")
    _ = (template_snapshot / "imatrix.dat").write_bytes(b"imatrix")
    _ = (template_snapshot / "mmproj.bin").write_bytes(b"mmproj")

    fake_params = QuantParams(
        tensor_types=["blk\\.(\\d+)\\.attn_q\\.weight=Q4_K"],
        default_type="Q4_K",
        quant_type_counts={"Q4_K": 10},
    )

    with (
        patch("gguf_clone.main.check_deps", return_value=[]),
        patch("gguf_clone.main.check_gguf_support", return_value=None),
        patch("gguf_clone.stages.extract_template.resolve_source_snapshot", return_value=template_snapshot),
        patch("gguf_clone.stages.extract_template.build_params", return_value=fake_params),
        patch(
            "gguf_clone.stages.extract_template.extract_template_metadata",
            return_value={"tokenizer.chat_template": "{{ chat }}"},
        ),
    ):
        result = run_extract_template(config, arts, overwrite_behavior="overwrite")

    assert result == 0
    payload = load_params(arts.params_gguf("Q4_K"))
    assert payload is not None
    assert payload.imatrix == "params/imatrix.dat"
    assert payload.template_metadata == {"tokenizer.chat_template": "{{ chat }}"}
    assert payload.staged_files == ["mmproj.bin"]
    assert payload.template_gguf == "model-Q4_K.gguf"
    assert (arts.template_files_dir / "mmproj.bin").exists()


def test_run_extract_template_missing_section(tmp_path: Path) -> None:
    config = _v2_config(tmp_path)
    arts = _v2_artifacts(tmp_path)

    with (
        patch("gguf_clone.main.check_deps", return_value=[]),
        patch("gguf_clone.main.check_gguf_support", return_value=None),
    ):
        result = run_extract_template(config, arts)

    assert result == 1


def test_run_quantize_gguf_with_target_convert(tmp_path: Path) -> None:
    config = _v2_config(
        tmp_path,
        quantize_gguf={
            "target_convert": True,
            "imatrix": None,
            "copy_metadata": [],
            "copy_files": [],
            "apply_metadata": {},
        },
    )
    arts = _v2_artifacts(tmp_path)
    arts.mkdir_all()

    payload = ParamsPayload(
        tensor_types=["tensor.weight=Q4_K"],
        default_type="Q4_K",
        imatrix="",
    )
    from gguf_clone.params import save_params_payload

    save_params_payload(payload, arts.params_gguf("Q4_K"))

    target_snapshot = tmp_path / "target_snap"
    target_snapshot.mkdir()

    def fake_convert(
        _model_path: Path,
        *,
        output_dir: Path,
        outfile_name: str | None,
        **_kwargs: object,
    ) -> int:
        assert outfile_name is not None
        _ = (output_dir / outfile_name).write_bytes(b"converted")
        return 0

    def fake_quantize(
        _input: Path,
        output_path: Path,
        **_kwargs: object,
    ) -> int:
        _ = output_path.write_bytes(b"quantized")
        return 0

    calls: list[str] = []

    def tracking_resolve(ref: SourceRef, **_kw: object) -> Path:
        if ref.repo == "org/target":
            calls.append("resolve_target")
            return target_snapshot
        raise AssertionError("Template resolution should not be needed here")

    with (
        patch("gguf_clone.main.check_deps", return_value=[]),
        patch("gguf_clone.main.check_gguf_support", return_value=None),
        patch("gguf_clone.stages.quantize_gguf.resolve_tools", return_value=_tool_paths()),
        patch("gguf_clone.stages.quantize_gguf.resolve_source_snapshot", side_effect=tracking_resolve),
        patch("gguf_clone.stages.quantize_gguf.convert_target", side_effect=fake_convert),
        patch("gguf_clone.stages.quantize_gguf.quantize_gguf", side_effect=fake_quantize),
    ):
        result = run_quantize_gguf(config, arts, overwrite_behavior="overwrite")

    assert result == 0
    assert calls == ["resolve_target"]
    assert arts.quantized_gguf("Q4_K").exists()


def test_run_quantize_gguf_missing_section(tmp_path: Path) -> None:
    config = _v2_config(tmp_path)
    arts = _v2_artifacts(tmp_path)

    with (
        patch("gguf_clone.main.check_deps", return_value=[]),
        patch("gguf_clone.main.check_gguf_support", return_value=None),
    ):
        result = run_quantize_gguf(config, arts)

    assert result == 1


def test_run_quantize_gguf_explicit_target_gguf(tmp_path: Path) -> None:
    target_gguf = tmp_path / "my-model.gguf"
    _ = target_gguf.write_bytes(b"existing-gguf")

    config = _v2_config(
        tmp_path,
        quantize_gguf={
            "target_gguf": str(target_gguf),
            "target_convert": False,
            "imatrix": None,
            "copy_metadata": [],
            "copy_files": [],
            "apply_metadata": {},
        },
    )
    arts = _v2_artifacts(tmp_path)
    arts.mkdir_all()

    payload = ParamsPayload(
        tensor_types=["tensor.weight=Q4_K"],
        default_type="Q4_K",
        imatrix="",
    )
    from gguf_clone.params import save_params_payload

    save_params_payload(payload, arts.params_gguf("Q4_K"))

    quantized_input: list[Path] = []

    def fake_quantize(
        input_path: Path,
        output_path: Path,
        **_kwargs: object,
    ) -> int:
        quantized_input.append(input_path)
        _ = output_path.write_bytes(b"quantized")
        return 0

    with (
        patch("gguf_clone.main.check_deps", return_value=[]),
        patch("gguf_clone.main.check_gguf_support", return_value=None),
        patch("gguf_clone.stages.quantize_gguf.resolve_tools", return_value=_tool_paths()),
        patch("gguf_clone.stages.quantize_gguf.resolve_source_snapshot") as resolve_mock,
        patch("gguf_clone.stages.quantize_gguf.quantize_gguf", side_effect=fake_quantize),
    ):
        result = run_quantize_gguf(config, arts, overwrite_behavior="overwrite")

    assert result == 0
    resolve_mock.assert_not_called()
    assert quantized_input == [target_gguf]
    assert arts.quantized_gguf("Q4_K").exists()


def test_run_quantize_gguf_no_params_files(tmp_path: Path) -> None:
    config = _v2_config(
        tmp_path,
        quantize_gguf={
            "target_convert": True,
            "imatrix": None,
            "copy_metadata": [],
            "copy_files": [],
            "apply_metadata": {},
        },
    )
    arts = _v2_artifacts(tmp_path)
    arts.mkdir_all()

    target_snapshot = tmp_path / "target_snap"
    target_snapshot.mkdir()

    def fake_convert(
        _model_path: Path,
        *,
        output_dir: Path,
        outfile_name: str | None,
        **_kwargs: object,
    ) -> int:
        assert outfile_name is not None
        _ = (output_dir / outfile_name).write_bytes(b"converted")
        return 0

    with (
        patch("gguf_clone.main.check_deps", return_value=[]),
        patch("gguf_clone.main.check_gguf_support", return_value=None),
        patch("gguf_clone.stages.quantize_gguf.resolve_tools", return_value=_tool_paths()),
        patch("gguf_clone.stages.quantize_gguf.resolve_source_snapshot", return_value=target_snapshot),
        patch("gguf_clone.stages.quantize_gguf.convert_target", side_effect=fake_convert),
    ):
        result = run_quantize_gguf(config, arts, overwrite_behavior="overwrite")

    assert result == 1


def test_run_quantize_gguf_requires_extracted_metadata(tmp_path: Path) -> None:
    config = _v2_config(
        tmp_path,
        quantize_gguf={
            "target_convert": True,
            "imatrix": None,
            "copy_metadata": ["tokenizer.chat_template"],
            "copy_files": [],
            "apply_metadata": {},
        },
    )
    arts = _v2_artifacts(tmp_path)
    arts.mkdir_all()

    payload = ParamsPayload(
        tensor_types=["tensor.weight=Q4_K"],
        default_type="Q4_K",
        imatrix="",
    )
    from gguf_clone.params import save_params_payload

    save_params_payload(payload, arts.params_gguf("Q4_K"))

    target_snapshot = tmp_path / "target_snap"
    target_snapshot.mkdir()

    def fake_convert(
        _model_path: Path,
        *,
        output_dir: Path,
        outfile_name: str | None,
        **_kwargs: object,
    ) -> int:
        assert outfile_name is not None
        _ = (output_dir / outfile_name).write_bytes(b"converted")
        return 0

    with (
        patch("gguf_clone.main.check_deps", return_value=[]),
        patch("gguf_clone.main.check_gguf_support", return_value=None),
        patch("gguf_clone.stages.quantize_gguf.resolve_tools", return_value=_tool_paths()),
        patch("gguf_clone.stages.quantize_gguf.resolve_source_snapshot", return_value=target_snapshot),
        patch("gguf_clone.stages.quantize_gguf.convert_target", side_effect=fake_convert),
        patch("gguf_clone.stages.quantize_gguf.quantize_gguf") as quantize_mock,
    ):
        result = run_quantize_gguf(config, arts, overwrite_behavior="overwrite")

    assert result == 1
    quantize_mock.assert_not_called()


def test_run_quantize_gguf_applies_extracted_metadata(tmp_path: Path) -> None:
    target_gguf = tmp_path / "target.gguf"
    _ = target_gguf.write_bytes(b"target")

    config = _v2_config(
        tmp_path,
        quantize_gguf={
            "target_gguf": str(target_gguf),
            "target_convert": False,
            "imatrix": None,
            "copy_metadata": ["tokenizer.chat_template"],
            "copy_files": [],
            "apply_metadata": {"general.quantized_by": "tester"},
        },
    )
    arts = _v2_artifacts(tmp_path)
    arts.mkdir_all()

    payload = ParamsPayload(
        tensor_types=["tensor.weight=Q4_K"],
        default_type="Q4_K",
        imatrix="",
        template_metadata={"tokenizer.chat_template": "{{ chat }}"},
    )
    from gguf_clone.params import save_params_payload

    save_params_payload(payload, arts.params_gguf("Q4_K"))

    def fake_quantize(
        _input: Path,
        output_path: Path,
        **_kwargs: object,
    ) -> int:
        _ = output_path.write_bytes(b"quantized")
        return 0

    with (
        patch("gguf_clone.main.check_deps", return_value=[]),
        patch("gguf_clone.main.check_gguf_support", return_value=None),
        patch("gguf_clone.stages.quantize_gguf.resolve_tools", return_value=_tool_paths()),
        patch("gguf_clone.stages.quantize_gguf.quantize_gguf", side_effect=fake_quantize),
        patch("gguf_clone.stages.quantize_gguf.apply_metadata", return_value=0) as apply_mock,
    ):
        result = run_quantize_gguf(config, arts, overwrite_behavior="overwrite")

    assert result == 0
    assert apply_mock.call_count == 2
    assert apply_mock.call_args_list[0].args[1] == {
        "tokenizer.chat_template": "{{ chat }}",
    }
    assert apply_mock.call_args_list[1].args[1] == {
        "general.quantized_by": "tester",
    }


def test_run_quantize_gguf_copies_staged_files(tmp_path: Path) -> None:
    target_gguf = tmp_path / "target.gguf"
    _ = target_gguf.write_bytes(b"target")

    config = _v2_config(
        tmp_path,
        quantize_gguf={
            "target_gguf": str(target_gguf),
            "target_convert": False,
            "imatrix": None,
            "copy_metadata": [],
            "copy_files": ["*mmproj*"],
            "apply_metadata": {},
        },
    )
    arts = _v2_artifacts(tmp_path)
    arts.mkdir_all()

    _ = (arts.template_files_dir / "mmproj.bin").write_bytes(b"staged")

    payload = ParamsPayload(
        tensor_types=["tensor.weight=Q4_K"],
        default_type="Q4_K",
        imatrix="",
        staged_files=["mmproj.bin"],
    )
    from gguf_clone.params import save_params_payload

    save_params_payload(payload, arts.params_gguf("Q4_K"))

    def fake_quantize(
        _input: Path,
        output_path: Path,
        **_kwargs: object,
    ) -> int:
        _ = output_path.write_bytes(b"quantized")
        return 0

    with (
        patch("gguf_clone.main.check_deps", return_value=[]),
        patch("gguf_clone.main.check_gguf_support", return_value=None),
        patch("gguf_clone.stages.quantize_gguf.resolve_tools", return_value=_tool_paths()),
        patch("gguf_clone.stages.quantize_gguf.quantize_gguf", side_effect=fake_quantize),
    ):
        result = run_quantize_gguf(config, arts, overwrite_behavior="overwrite")

    assert result == 0
    assert (arts.quantized_gguf_dir / "mmproj.bin").exists()


def test_run_pipeline_missing_config_file(tmp_path: Path) -> None:
    result = run_pipeline(tmp_path / "does-not-exist.yml")
    assert result == 1


def test_run_pipeline_executes_stages_in_order(tmp_path: Path) -> None:
    import yaml

    config_data: dict[str, object] = {
        "version": 2,
        "source": {
            "template": "org/template",
            "target": "org/target",
        },
        "output_dir": "output",
        "extract_template": {
            "ggufs": ["*.gguf"],
        },
        "quantize_gguf": {
            "imatrix": None,
            "copy_metadata": [],
            "copy_files": [],
            "apply_metadata": {},
        },
    }
    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    calls: list[str] = []

    def fake_extract(config: V2RunConfig, arts: Artifacts, **_kw: object) -> int:
        del config, arts
        calls.append("extract_template")
        return 0

    def fake_quantize(config: V2RunConfig, arts: Artifacts, **_kw: object) -> int:
        del config, arts
        calls.append("quantize_gguf")
        return 0

    with (
        patch("gguf_clone.main.run_extract_template", side_effect=fake_extract),
        patch("gguf_clone.main.run_quantize_gguf", side_effect=fake_quantize),
    ):
        result = run_pipeline(config_file)

    assert result == 0
    assert calls == ["extract_template", "quantize_gguf"]


def test_run_pipeline_skips_absent_stages(tmp_path: Path) -> None:
    import yaml

    config_data = {
        "version": 2,
        "source": {
            "template": "org/template",
            "target": "org/target",
        },
        "extract_template": {
            "ggufs": ["*.gguf"],
        },
    }
    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    calls: list[str] = []

    def fake_extract(config: V2RunConfig, arts: Artifacts, **_kw: object) -> int:
        del config, arts
        calls.append("extract_template")
        return 0

    with (
        patch("gguf_clone.main.run_extract_template", side_effect=fake_extract),
        patch("gguf_clone.main.run_quantize_gguf") as qg_mock,
    ):
        result = run_pipeline(config_file)

    assert result == 0
    assert calls == ["extract_template"]
    qg_mock.assert_not_called()


def test_run_pipeline_no_stages_errors(tmp_path: Path) -> None:
    import yaml

    config_data = {
        "version": 2,
        "source": {
            "template": "org/template",
            "target": "org/target",
        },
    }
    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    result = run_pipeline(config_file)
    assert result == 1


def test_run_pipeline_stops_on_stage_failure(tmp_path: Path) -> None:
    import yaml

    config_data: dict[str, object] = {
        "version": 2,
        "source": {
            "template": "org/template",
            "target": "org/target",
        },
        "extract_template": {
            "ggufs": ["*.gguf"],
        },
        "quantize_gguf": {
            "imatrix": None,
            "apply_metadata": {},
        },
    }
    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    def fake_extract_fail(config: V2RunConfig, arts: Artifacts, **_kw: object) -> int:
        del config, arts
        return 1

    with (
        patch("gguf_clone.main.run_extract_template", side_effect=fake_extract_fail),
        patch("gguf_clone.main.run_quantize_gguf") as qg_mock,
    ):
        result = run_pipeline(config_file)

    assert result == 1
    qg_mock.assert_not_called()
