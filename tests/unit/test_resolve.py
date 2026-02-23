from pathlib import Path

from pytest import CaptureFixture

from gguf_clone.resolve import filter_paths, match_pattern, resolve_models


def test_filter_paths_logic() -> None:
    base = Path("/base")
    candidates = [
        base / "a.gguf",
        base / "b.txt",
        base / "sub/c.gguf",
        base / "sub/d.json",
    ]

    # Test allow patterns
    filtered = filter_paths(base, candidates, ["*.gguf"], None)
    assert len(filtered) == 2
    assert base / "a.gguf" in filtered
    assert base / "sub/c.gguf" in filtered

    # Test ignore patterns
    filtered = filter_paths(base, candidates, None, ["*.txt", "*.json"])
    assert len(filtered) == 2
    assert base / "a.gguf" in filtered
    assert base / "sub/c.gguf" in filtered

    # Test both
    filtered = filter_paths(base, candidates, ["*"], ["sub/*"])
    assert len(filtered) == 2
    assert base / "a.gguf" in filtered
    assert base / "b.txt" in filtered


def test_match_pattern_single_success(capsys: CaptureFixture[str]) -> None:
    base = Path("/base")
    candidates = [base / "target.gguf", base / "other.txt"]

    result = match_pattern(base, candidates, "*.gguf", "test", require_single=True)
    assert result == [base / "target.gguf"]

    captured = capsys.readouterr()
    assert captured.out == ""


def test_match_pattern_single_none(capsys: CaptureFixture[str]) -> None:
    base = Path("/base")
    candidates = [base / "other.txt"]

    result = match_pattern(base, candidates, "*.gguf", "test", require_single=True)
    assert result is None

    captured = capsys.readouterr()
    assert "No matches found for test pattern: *.gguf" in captured.out


def test_match_pattern_single_ambiguous(capsys: CaptureFixture[str]) -> None:
    base = Path("/base")
    candidates = [base / "a.gguf", base / "b.gguf"]

    result = match_pattern(base, candidates, "*.gguf", "test", require_single=True)
    assert result is None

    captured = capsys.readouterr()
    assert "test pattern matched 2 files" in captured.out
    assert "a.gguf" in captured.out
    assert "b.gguf" in captured.out


def test_match_pattern_group_success(capsys: CaptureFixture[str]) -> None:
    base = Path("/base")
    candidates = [base / "target.gguf", base / "other.txt"]

    result = match_pattern(base, candidates, "*.gguf", "test")
    assert result == [base / "target.gguf"]

    captured = capsys.readouterr()
    assert captured.out == ""


def test_match_pattern_group_multiple(capsys: CaptureFixture[str]) -> None:
    base = Path("/base")
    candidates = [base / "a.gguf", base / "b.gguf"]

    result = match_pattern(base, candidates, "*.gguf", "test")
    assert result == [base / "a.gguf", base / "b.gguf"]

    captured = capsys.readouterr()
    assert captured.out == ""


def test_match_pattern_group_none(capsys: CaptureFixture[str]) -> None:
    base = Path("/base")
    candidates = [base / "other.txt"]

    result = match_pattern(base, candidates, "*.gguf", "test")
    assert result is None

    captured = capsys.readouterr()
    assert "No matches found for test pattern: *.gguf" in captured.out


def test_resolve_models_local_sources(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    _ = (template_dir / "model-Q4_K.gguf").write_bytes(b"gguf")
    _ = (template_dir / "imatrix.dat").write_bytes(b"imatrix")

    target_dir = tmp_path / "target"
    target_dir.mkdir()

    resolved = resolve_models(
        template_repo=None,
        template_path=template_dir,
        template_gguf_patterns=["*.gguf"],
        template_imatrix_pattern="imatrix.dat",
        template_copy_files=[],
        target_repo=None,
        target_path=target_dir,
        target_exclude_files=["*.md"],
    )

    assert resolved is not None
    assert resolved.template_snapshot == template_dir
    assert resolved.template_imatrix == template_dir / "imatrix.dat"
    assert resolved.template_ggufs == [[template_dir / "model-Q4_K.gguf"]]
    assert resolved.target_snapshot == target_dir

    captured = capsys.readouterr()
    assert "Ignoring target.exclude_files for local target path." in captured.out


def test_resolve_models_rejects_non_gguf_target_file(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    _ = (template_dir / "model-Q4_K.gguf").write_bytes(b"gguf")
    _ = (template_dir / "imatrix.dat").write_bytes(b"imatrix")

    bad_target = tmp_path / "target.bin"
    _ = bad_target.write_bytes(b"bad")

    resolved = resolve_models(
        template_repo=None,
        template_path=template_dir,
        template_gguf_patterns=["*.gguf"],
        template_imatrix_pattern="imatrix.dat",
        template_copy_files=[],
        target_repo=None,
        target_path=bad_target,
        target_exclude_files=None,
    )

    assert resolved is None
    captured = capsys.readouterr()
    assert "Target path must be a directory or a .gguf file." in captured.out
