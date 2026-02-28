from pathlib import Path

import yaml
from pytest import CaptureFixture

from gguf_clone.config import RunConfig, load_config


def test_load_full_config(tmp_path: Path) -> None:
    config_data = {
        "version": 2,
        "source": {
            "template": "unsloth/Qwen3-0.6B-GGUF",
            "target": "Qwen/Qwen3-0.6B",
        },
        "output_dir": "output",
        "extract_template": {
            "ggufs": ["*UD-IQ1_M*.gguf", "*UD-Q2_K_XL*.gguf"],
        },
        "extract_template_mlx": {
            "trust_remote_code": True,
        },
        "quantize_gguf": {
            "target_gguf": None,
            "target_convert": True,
            "imatrix": "*imatrix*",
            "output_max_size": "50G",
            "copy_metadata": ["tokenizer.chat_template"],
            "copy_files": ["*mmproj*"],
            "apply_metadata": {
                "general.quantized_by": "test-user",
            },
        },
        "quantize_mlx": {
            "group_size": 128,
            "trust_remote_code": True,
        },
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is not None
    assert isinstance(config, RunConfig)
    # Source resolves as HF repos (paths don't exist locally)
    assert config.template.repo == "unsloth/Qwen3-0.6B-GGUF"
    assert config.template.path is None
    assert config.target.repo == "Qwen/Qwen3-0.6B"
    assert config.target.path is None
    assert config.output_dir == tmp_path / "output"

    # extract_template
    assert config.extract_template is not None
    assert config.extract_template.ggufs == ["*UD-IQ1_M*.gguf", "*UD-Q2_K_XL*.gguf"]

    # extract_template_mlx
    assert config.extract_template_mlx is not None
    assert config.extract_template_mlx.trust_remote_code is True
    assert config.extract_template_mlx.package == "mlx-lm"

    # quantize_gguf
    assert config.quantize_gguf is not None
    assert config.quantize_gguf.target_gguf is None
    assert config.quantize_gguf.target_convert is True
    assert config.quantize_gguf.imatrix == "*imatrix*"
    assert config.quantize_gguf.output_max_size == "50G"
    assert config.quantize_gguf.copy_metadata == ["tokenizer.chat_template"]
    assert config.quantize_gguf.copy_files == ["*mmproj*"]
    assert config.quantize_gguf.apply_metadata == {
        "general.quantized_by": "test-user",
    }

    # quantize_mlx
    assert config.quantize_mlx is not None
    assert config.quantize_mlx.group_size == 128
    assert config.quantize_mlx.trust_remote_code is True


def test_omitted_stages_are_none(tmp_path: Path) -> None:
    config_data = {
        "version": 2,
        "source": {
            "template": "unsloth/Qwen3-0.6B-GGUF",
            "target": "Qwen/Qwen3-0.6B",
        },
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is not None
    assert config.extract_template is None
    assert config.extract_template_mlx is None
    assert config.quantize_gguf is None
    assert config.quantize_mlx is None


def test_single_stage_only(tmp_path: Path) -> None:
    config_data = {
        "version": 2,
        "source": {
            "template": "unsloth/Qwen3-0.6B-GGUF",
            "target": "Qwen/Qwen3-0.6B",
        },
        "extract_template": {
            "ggufs": "*UD-IQ1_M*.gguf",
        },
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is not None
    assert config.extract_template is not None
    assert config.extract_template.ggufs == ["*UD-IQ1_M*.gguf"]
    assert config.quantize_gguf is None
    assert config.quantize_mlx is None


def test_defaults(tmp_path: Path) -> None:
    config_data = {
        "version": 2,
        "source": {
            "template": "unsloth/Qwen3-0.6B-GGUF",
            "target": "Qwen/Qwen3-0.6B",
        },
        "quantize_gguf": {
            "imatrix": "*imatrix*",
        },
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is not None
    assert config.output_dir == tmp_path / "output"

    qg = config.quantize_gguf
    assert qg is not None
    assert qg.target_gguf is None
    assert qg.target_convert is True
    assert qg.output_max_size == "50G"
    assert qg.copy_metadata == []
    assert qg.copy_files == []
    assert qg.apply_metadata == {
        "general.quantized_by": "https://github.com/spicyneuron/gguf-clone"
    }


def test_local_path_source(tmp_path: Path) -> None:
    template_dir = tmp_path / "models" / "template"
    target_dir = tmp_path / "models" / "target"
    template_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    # Template dir must contain at least one .gguf file
    _ = (template_dir / "model.gguf").write_bytes(b"")

    config_data = {
        "version": 2,
        "source": {
            "template": "models/template",
            "target": "models/target",
        },
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is not None
    assert config.template.repo is None
    assert config.template.path == template_dir
    assert config.target.repo is None
    assert config.target.path == target_dir


def test_nonexistent_path_falls_back_to_repo(tmp_path: Path) -> None:
    config_data = {
        "version": 2,
        "source": {
            "template": "org/some-model",
            "target": "org/other-model",
        },
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is not None
    assert config.template.repo == "org/some-model"
    assert config.template.path is None
    assert config.target.repo == "org/other-model"
    assert config.target.path is None


def test_missing_source(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    config_data = {
        "version": 2,
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is None
    captured = capsys.readouterr()
    assert "source" in captured.out


def test_missing_config_file(capsys: CaptureFixture[str]) -> None:
    config = load_config(Path("/does-not-exist/config.yml"))
    assert config is None
    captured = capsys.readouterr()
    assert "Config file not found" in captured.out


def test_missing_source_template(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    config_data = {
        "version": 2,
        "source": {"target": "Qwen/Qwen3-0.6B"},
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is None
    captured = capsys.readouterr()
    assert "template" in captured.out


def test_unsupported_version(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    config_data = {
        "version": 1,
        "source": {
            "template": "org/model",
            "target": "org/target",
        },
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is None
    captured = capsys.readouterr()
    assert "Unsupported config version" in captured.out


def test_extract_template_string_ggufs(tmp_path: Path) -> None:
    """Single string ggufs should be coerced to a list."""
    config_data = {
        "version": 2,
        "source": {
            "template": "org/model",
            "target": "org/target",
        },
        "extract_template": {
            "ggufs": "*Q4_K*.gguf",
        },
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is not None
    assert config.extract_template is not None
    assert config.extract_template.ggufs == ["*Q4_K*.gguf"]


def test_quantize_gguf_no_imatrix(tmp_path: Path) -> None:
    """Setting imatrix to null should skip imatrix."""
    config_data = {
        "version": 2,
        "source": {
            "template": "org/model",
            "target": "org/target",
        },
        "quantize_gguf": {
            "imatrix": None,
        },
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is not None
    assert config.quantize_gguf is not None
    assert config.quantize_gguf.imatrix is None


def test_quantize_gguf_clear_apply_metadata(tmp_path: Path) -> None:
    config_data: dict[str, object] = {
        "version": 2,
        "source": {
            "template": "org/model",
            "target": "org/target",
        },
        "quantize_gguf": {
            "apply_metadata": {},
        },
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is not None
    assert config.quantize_gguf is not None
    assert config.quantize_gguf.apply_metadata == {}


def test_extract_template_mlx_defaults(tmp_path: Path) -> None:
    config_data = {
        "version": 2,
        "source": {
            "template": "org/model",
            "target": "org/target",
        },
        "extract_template_mlx": {},
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is not None
    assert config.extract_template_mlx is not None
    assert config.extract_template_mlx.package == "mlx-lm"
    assert config.extract_template_mlx.trust_remote_code is False


def test_quantize_mlx_defaults(tmp_path: Path) -> None:
    config_data = {
        "version": 2,
        "source": {
            "template": "org/model",
            "target": "org/target",
        },
        "quantize_mlx": {},
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is not None
    assert config.quantize_mlx is not None
    assert config.quantize_mlx.group_size == 64
    assert config.quantize_mlx.trust_remote_code is False


def test_output_dir_relative_to_config(tmp_path: Path) -> None:
    config_data = {
        "version": 2,
        "source": {
            "template": "org/model",
            "target": "org/target",
        },
        "output_dir": "my_output",
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is not None
    assert config.output_dir == tmp_path / "my_output"


def test_local_template_dir_must_contain_ggufs(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    empty_dir = tmp_path / "empty_template"
    empty_dir.mkdir()

    config_data = {
        "version": 2,
        "source": {
            "template": "empty_template",
            "target": "org/target",
        },
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is None
    captured = capsys.readouterr()
    assert "no .gguf files" in captured.out.lower()


def test_source_file_not_treated_as_local(tmp_path: Path) -> None:
    """A file (not a directory) should fall back to HF repo."""
    _ = (tmp_path / "not-a-dir").write_bytes(b"")

    config_data = {
        "version": 2,
        "source": {
            "template": "not-a-dir",
            "target": "not-a-dir",
        },
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is not None
    # Both should fall back to HF repo since they're files, not dirs
    assert config.template.repo == "not-a-dir"
    assert config.template.path is None
    assert config.target.repo == "not-a-dir"
    assert config.target.path is None
