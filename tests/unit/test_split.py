from gguf_clone.split import parse_split_size, split_needed


def test_parse_split_size_units() -> None:
    assert parse_split_size("50G") == 50 * 1024 * 1024 * 1024
    assert parse_split_size("512M") == 512 * 1024 * 1024
    assert parse_split_size("1024") == 1024
    assert parse_split_size(" 5g ") == 5 * 1024 * 1024 * 1024


def test_parse_split_size_invalid() -> None:
    assert parse_split_size("") is None
    assert parse_split_size("G50") is None
    assert parse_split_size("10T") is None


def test_split_needed() -> None:
    limit = 5 * 1024 * 1024
    assert split_needed(limit, "5M") is False
    assert split_needed(limit + 1, "5M") is True
    assert split_needed(limit, "bad") is None
