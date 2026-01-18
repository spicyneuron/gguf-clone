import sys
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

from gguf_clone.convert import convert_target
from gguf_clone.quantize import build_quantize_options, quantize_gguf
from gguf_clone.split import build_split_options, split_gguf


def test_build_quantize_options() -> None:
    tensor_types = ["layer.1.weight=Q4_K", "layer.2.weight=Q6_K"]
    imatrix = "imatrix.dat"

    options = build_quantize_options(tensor_types, imatrix)

    expected = [
        "--imatrix",
        "imatrix.dat",
        "--tensor-type",
        "layer.1.weight=Q4_K",
        "--tensor-type",
        "layer.2.weight=Q6_K",
    ]
    assert options == expected


def test_build_split_options() -> None:
    options = build_split_options("50G")

    assert options == ["--split", "--split-max-size", "50G"]


@patch("gguf_clone.convert.run_command")
def test_convert_target_command(mock_run: MagicMock, tmp_path: Path) -> None:
    model_path = Path("/model")
    output_dir = tmp_path / "out"
    convert_script = Path("/scripts/convert.py")
    work_dir = tmp_path / "work"

    _ = convert_target(
        model_path,
        output_dir=output_dir,
        outfile_name="converted.gguf",
        existing_glob="converted.gguf",
        convert_script=convert_script,
        cwd=work_dir,
    )

    mock_run.assert_called_once()
    cmd = cast(list[str], mock_run.call_args[0][0])
    kwargs = cast(dict[str, object], mock_run.call_args[1])

    assert cmd[0] == sys.executable
    assert cmd[1] == str(convert_script)
    assert "--outtype" in cmd
    assert "auto" in cmd
    assert "--outfile" in cmd
    assert str(output_dir / "converted.gguf") in cmd
    assert str(model_path) in cmd
    assert kwargs["cwd"] == work_dir
    assert isinstance(kwargs["env"], dict)


@patch("gguf_clone.quantize.run_command")
def test_quantize_gguf_command(mock_run: MagicMock, tmp_path: Path) -> None:
    input_path = Path("in.gguf")
    output_path = Path("out.gguf")
    llama_quantize = Path("/bin/llama-quantize")
    tensor_types = ["t=Q4_0"]
    work_dir = tmp_path / "work"

    _ = quantize_gguf(
        input_path,
        output_path,
        tensor_types=tensor_types,
        default_type="Q8_0",
        imatrix="imatrix.dat",
        llama_quantize=llama_quantize,
        cwd=work_dir,
    )

    mock_run.assert_called_once()
    cmd = cast(list[str], mock_run.call_args[0][0])
    kwargs = cast(dict[str, object], mock_run.call_args[1])

    assert cmd[0] == str(llama_quantize)
    assert "--imatrix" in cmd
    assert "imatrix.dat" in cmd
    assert "--tensor-type" in cmd
    assert "t=Q4_0" in cmd
    assert str(input_path) in cmd
    assert str(output_path) in cmd
    assert "Q8_0" in cmd
    assert kwargs["cwd"] == work_dir


@patch("gguf_clone.split.run_command")
def test_split_gguf_command(mock_run: MagicMock, tmp_path: Path) -> None:
    input_path = Path("in.gguf")
    output_path = Path("out.gguf")
    llama_gguf_split = Path("/bin/llama-gguf-split")
    work_dir = tmp_path / "work"

    _ = split_gguf(
        input_path,
        output_path,
        max_size="50G",
        llama_gguf_split=llama_gguf_split,
        cwd=work_dir,
    )

    mock_run.assert_called_once()
    cmd = cast(list[str], mock_run.call_args[0][0])
    kwargs = cast(dict[str, object], mock_run.call_args[1])

    assert cmd[0] == str(llama_gguf_split)
    assert "--split" in cmd
    assert "--split-max-size" in cmd
    assert "50G" in cmd
    assert str(input_path) in cmd
    assert str(output_path) in cmd
    assert kwargs["cwd"] == work_dir
