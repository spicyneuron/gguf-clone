from pathlib import Path

from gguf_clone.artifacts import Artifacts, quant_label_from_stem, source_desc, source_slug
from gguf_clone.config import SourceRef


def test_source_slug_repo() -> None:
    ref = SourceRef(repo="unsloth/Qwen3-0.6B-GGUF", path=None)
    assert source_slug(ref) == "unsloth-Qwen3-0.6B-GGUF"


def test_source_slug_path() -> None:
    ref = SourceRef(repo=None, path=Path("/models/my-model"))
    assert source_slug(ref) == "my-model"


def test_source_slug_path_with_suffix() -> None:
    ref = SourceRef(repo=None, path=Path("/models/model.gguf"))
    assert source_slug(ref) == "model"


def test_source_slug_fallback() -> None:
    ref = SourceRef(repo=None, path=None)
    assert source_slug(ref) == "model"


def test_source_desc_repo() -> None:
    ref = SourceRef(repo="org/model", path=None)
    assert source_desc(ref) == "repo:org/model"


def test_source_desc_path() -> None:
    ref = SourceRef(repo=None, path=Path("/some/path"))
    assert source_desc(ref) == "path:/some/path"


def test_artifacts_from_config() -> None:
    template = SourceRef(repo="unsloth/Qwen3-0.6B-GGUF", path=None)
    target = SourceRef(repo="Qwen/Qwen3-0.6B", path=None)
    arts = Artifacts.from_config(Path("/out"), template, target)

    assert arts.template_slug == "unsloth-Qwen3-0.6B-GGUF"
    assert arts.target_slug == "Qwen-Qwen3-0.6B"


def test_artifacts_directory_layout() -> None:
    arts = Artifacts(
        output_dir=Path("/out"),
        template_slug="tmpl",
        target_slug="tgt",
    )
    assert arts.converted_dir == Path("/out/converted")
    assert arts.params_dir == Path("/out/params")
    assert arts.template_files_dir == Path("/out/template_files")
    assert arts.quantized_gguf_dir == Path("/out/quantized/gguf")
    assert arts.quantized_mlx_dir == Path("/out/quantized/mlx")


def test_artifacts_file_paths() -> None:
    arts = Artifacts(
        output_dir=Path("/out"),
        template_slug="unsloth-Qwen3-0.6B-GGUF",
        target_slug="Qwen-Qwen3-0.6B",
    )
    assert arts.converted_gguf() == Path("/out/converted/Qwen-Qwen3-0.6B.gguf")
    assert arts.params_gguf("UD-IQ1_M") == Path(
        "/out/params/unsloth-Qwen3-0.6B-GGUF-UD-IQ1_M-gguf.json"
    )
    assert arts.params_mlx("UD-IQ1_M") == Path(
        "/out/params/unsloth-Qwen3-0.6B-GGUF-UD-IQ1_M-mlx.json"
    )
    assert arts.quantized_gguf("UD-IQ1_M") == Path(
        "/out/quantized/gguf/Qwen-Qwen3-0.6B-UD-IQ1_M.gguf"
    )
    assert arts.quantized_mlx("UD-IQ1_M") == Path(
        "/out/quantized/mlx/UD-IQ1_M"
    )


def test_artifacts_mkdir_all(tmp_path: Path) -> None:
    arts = Artifacts(
        output_dir=tmp_path / "out",
        template_slug="t",
        target_slug="t",
    )
    arts.mkdir_all()
    assert arts.converted_dir.is_dir()
    assert arts.params_dir.is_dir()
    assert arts.template_files_dir.is_dir()
    assert arts.quantized_gguf_dir.is_dir()
    assert arts.quantized_mlx_dir.is_dir()


def test_quant_label_from_stem_ud() -> None:
    assert quant_label_from_stem("model-UD-IQ2_XXS") == "UD-IQ2_XXS"


def test_quant_label_from_stem_standard() -> None:
    assert quant_label_from_stem("model-Q4_K_M") == "Q4_K_M"


def test_quant_label_from_stem_float() -> None:
    assert quant_label_from_stem("model-F16") == "F16"


def test_quant_label_from_stem_unknown() -> None:
    assert quant_label_from_stem("model-unknown") is None
