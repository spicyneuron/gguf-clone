from pathlib import Path

import yaml
from pytest import CaptureFixture

from gguf_clone.main import RunConfig, load_config


def test_load_config_valid(tmp_path: Path) -> None:
    config_data = {
        "template": {
            "repo": "unsloth/Qwen3-0.6B-GGUF",
            "imatrix": "*imatrix*",
            "ggufs": ["*UD-IQ1_M.gguf"],
            "copy_metadata": ["tokenizer.chat_template"],
            "copy_files": ["*mmproj*"],
        },
        "target": {"repo": "unsloth/Qwen3-0.6B"},
        "output": {
            "prefix": "test_prefix",
            "split": "25G",
            "converted_dir": "custom_converted",
            "params_dir": "custom_params",
            "quantized_dir": "custom_quantized",
        },
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is not None
    assert isinstance(config, RunConfig)
    assert config.template_repo == "unsloth/Qwen3-0.6B-GGUF"
    assert config.template_imatrix_pattern == "*imatrix*"
    assert config.template_gguf_patterns == ["*UD-IQ1_M.gguf"]
    assert config.template_copy_metadata == ["tokenizer.chat_template"]
    assert config.template_copy_files == ["*mmproj*"]
    assert config.target_repo == "unsloth/Qwen3-0.6B"
    assert config.output_prefix == "test_prefix"
    assert config.output_split == "25G"
    assert config.output_converted_dir == "custom_converted"
    assert config.output_params_dir == "custom_params"
    assert config.output_quantized_dir == "custom_quantized"


def test_load_config_defaults(tmp_path: Path) -> None:
    # Test default output prefix, split, directories, and string ggufs
    config_data = {
        "template": {"repo": "repo", "imatrix": "imatrix", "ggufs": "*gguf"},
        "target": {"repo": "target"},
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is not None
    assert config.output_prefix == ""
    assert config.output_split == "50G"
    assert config.output_converted_dir == "converted"
    assert config.output_params_dir == "params"
    assert config.output_quantized_dir == "quantized"
    assert config.template_gguf_patterns == ["*gguf"]
    assert config.template_copy_metadata == []
    assert config.template_copy_files == []


def test_load_config_missing_keys(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    config_data = {
        "template": {
            "repo": "repo"
            # Missing ggufs, imatrix
        },
        "target": {"repo": "target"},
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is None
    captured = capsys.readouterr()
    assert "ggufs" in captured.out
    assert "Field required" in captured.out


def test_load_config_invalid_types(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    config_data = {
        "template": {
            "repo": 123,  # Invalid type
            "imatrix": "imatrix",
            "ggufs": ["*gguf"],
        },
        "target": {"repo": "target"},
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is None
    captured = capsys.readouterr()
    assert "template.repo" in captured.out or "repo" in captured.out
    assert "str" in captured.out.lower()


def test_load_config_partial_directory_overrides(tmp_path: Path) -> None:
    # Test that partial directory overrides work with defaults
    config_data = {
        "template": {"repo": "repo", "imatrix": "imatrix", "ggufs": "*gguf"},
        "target": {"repo": "target"},
        "output": {
            "quantized_dir": "my_outputs",
        },
    }

    config_file = tmp_path / "config.yml"
    _ = config_file.write_text(yaml.dump(config_data))

    config = load_config(config_file)

    assert config is not None
    assert config.output_converted_dir == "converted"  # Default
    assert config.output_params_dir == "params"  # Default
    assert config.output_quantized_dir == "my_outputs"  # Override
