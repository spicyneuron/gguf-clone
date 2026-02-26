from pathlib import Path

from pytest import CaptureFixture

from gguf_clone.resolve import filter_paths, match_pattern


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
