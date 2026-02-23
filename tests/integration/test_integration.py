from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gguf import GGUFReader

from gguf_clone import main
from gguf_clone.params import load_params
from gguf_clone.resolve import filter_paths, resolve_hf_repo
from gguf_clone.split import split_needed


@dataclass(frozen=True)
class TemplateGroup:
    pattern: str
    shards: list[Path]
    quant_label: str
    template_path: Path
    output_path: Path
    params_path: Path


def _read_metadata(path: Path, keys: list[str]) -> dict[str, object | None]:
    reader = GGUFReader(str(path), "r")
    values: dict[str, object | None] = {}
    for key in keys:
        field = reader.get_field(key)
        values[key] = field.contents() if field is not None else None
    return values


def _resolve_template_snapshot(config: main.RunConfig) -> Path:
    allow_patterns = [
        *config.template_gguf_patterns,
        config.template_imatrix_pattern,
        *config.template_copy_files,
    ]
    if config.template_repo is not None:
        return Path(
            resolve_hf_repo(
                config.template_repo,
                allow_patterns=allow_patterns,
                local_files_only=True,
            )
        )
    assert config.template_path is not None
    return config.template_path.parent if config.template_path.is_file() else config.template_path


def _collect_template_groups(
    template_snapshot: Path,
    patterns: list[str],
) -> list[tuple[str, list[Path]]] | None:
    gguf_candidates = [
        path for path in template_snapshot.rglob("*.gguf") if path.is_file()
    ]
    groups: list[tuple[str, list[Path]]] = []
    for pattern in patterns:
        matches = filter_paths(template_snapshot, gguf_candidates, [pattern], None)
        if not matches:
            return None
        groups.append((pattern, matches))
    return groups


def _collect_copy_files(
    template_snapshot: Path, patterns: list[str]
) -> list[Path] | None:
    if not patterns:
        return []
    candidates = [path for path in template_snapshot.rglob("*") if path.is_file()]
    matches: list[Path] = []
    for pattern in patterns:
        pattern_matches = filter_paths(template_snapshot, candidates, [pattern], None)
        if not pattern_matches:
            return None
        matches.extend(pattern_matches)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in matches:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _build_template_groups(
    config: main.RunConfig,
    template_snapshot: Path,
    *,
    params_dir: Path,
    quantized_dir: Path,
) -> list[TemplateGroup] | None:
    raw_groups = _collect_template_groups(
        template_snapshot, config.template_gguf_patterns
    )
    if raw_groups is None:
        return None

    template_repo_slug = main.source_slug(config.template_repo, config.template_path)
    target_repo_slug = main.source_slug(config.target_repo, config.target_path)
    prefix = main.prefix_slug(config.output_prefix)

    groups: list[TemplateGroup] = []
    seen_labels: set[str] = set()
    for pattern, shards in raw_groups:
        labels = {
            label for path in shards if (label := main.quant_label_from_stem(path.stem))
        }
        if len(labels) > 1:
            return None
        quant_label = next(iter(labels), None)
        if not quant_label:
            return None
        if quant_label in seen_labels:
            return None
        seen_labels.add(quant_label)

        params_path = params_dir / f"{template_repo_slug}-{quant_label}.json"
        output_name = f"{prefix}-{target_repo_slug}-{quant_label}"
        output_path = quantized_dir / f"{output_name}.gguf"
        groups.append(
            TemplateGroup(
                pattern=pattern,
                shards=shards,
                quant_label=quant_label,
                template_path=shards[0],
                output_path=output_path,
                params_path=params_path,
            )
        )
    return groups


def _split_outputs(output_path: Path, quant_label: str) -> list[Path]:
    split_dir = output_path.parent / quant_label
    if not split_dir.exists():
        return []
    return sorted(split_dir.glob(f"{output_path.stem}-*.gguf"))


def _summarize_value(value: object | None) -> str:
    if value is None:
        return "missing"
    if isinstance(value, str):
        trimmed = value.replace("\n", "\\n")
        if len(trimmed) > 80:
            return f"{trimmed[:77]}..."
        return trimmed
    return repr(value)


def test_integration_full_workflow() -> None:
    config_path = Path(__file__).parent / "config.yml"
    failures: list[str] = []

    def check(condition: bool, label: str, detail: str = "") -> None:
        if not condition:
            message = f"{label}"
            if detail:
                message += f" - {detail}"
            failures.append(message)

    result = main.run(config_path, overwrite_behavior="use")
    assert result == 0, "Main execution failed"

    config = main.load_config(config_path)
    assert config is not None, "Failed to load config"

    template_snapshot = _resolve_template_snapshot(config)

    work_dir = config_path.parent
    quantized_dir = work_dir / config.output_quantized_dir
    params_dir = work_dir / config.output_params_dir

    groups = _build_template_groups(
        config,
        template_snapshot,
        params_dir=params_dir,
        quantized_dir=quantized_dir,
    )
    assert groups is not None, "Failed to build template groups"
    assert len(groups) > 0, "No template groups found"

    check(quantized_dir.exists(), "quantized outputs directory exists")
    if not quantized_dir.exists():
        assert not failures, "\n".join(failures)

    for group in groups:
        check(group.params_path.exists(), "params file", group.params_path.name)
        if group.params_path.exists():
            payload = load_params(group.params_path)
            check(payload is not None, "params load", group.params_path.name)

    if config.template_copy_metadata:
        for group in groups:
            split_matches = _split_outputs(group.output_path, group.quant_label)
            output_path = group.output_path
            if not output_path.exists():
                output_path = split_matches[0] if split_matches else None
            if output_path is None:
                check(False, "metadata copy", f"{group.output_path.name} missing")
                continue
            template_values = _read_metadata(
                group.template_path, config.template_copy_metadata
            )
            output_values = _read_metadata(output_path, config.template_copy_metadata)
            for key in config.template_copy_metadata:
                template_value = template_values.get(key)
                output_value = output_values.get(key)
                ok = template_value is not None and template_value == output_value
                if not ok:
                    detail = (
                        f"{output_path.name} {key} "
                        f"expected={_summarize_value(template_value)} "
                        f"actual={_summarize_value(output_value)}"
                    )
                    check(ok, "metadata match", detail)

    copy_files = _collect_copy_files(template_snapshot, config.template_copy_files)
    assert copy_files is not None, "Failed to collect copy files"

    if copy_files:
        seen_names: set[str] = set()
        duplicate_names: set[str] = set()
        for path in copy_files:
            if path.name in seen_names:
                duplicate_names.add(path.name)
            seen_names.add(path.name)
        if duplicate_names:
            check(
                False,
                "copy_files",
                f"duplicate names: {', '.join(sorted(duplicate_names))}",
            )
        for path in copy_files:
            dest = quantized_dir / path.name
            check(dest.exists(), "copied file", dest.name)

    for group in groups:
        split_matches = _split_outputs(group.output_path, group.quant_label)
        output_size = None
        if group.output_path.exists():
            output_size = group.output_path.stat().st_size
        elif split_matches:
            output_size = sum(path.stat().st_size for path in split_matches)

        if output_size is None:
            check(
                False, "quantized outputs", f"missing output for {group.quant_label}"
            )
            continue

        needed = split_needed(output_size, config.output_split)
        check(needed is not None, "split check", f"valid split size {config.output_split}")
        if needed is None:
            continue

        if needed:
            check(
                bool(split_matches),
                "split outputs",
                f"{group.output_path.name} -> {len(split_matches)} file(s)",
            )
            check(
                (group.output_path.parent / group.quant_label).exists(),
                "split output dir",
                group.quant_label,
            )
            check(
                not group.output_path.exists(),
                "split source removed",
                group.output_path.name,
            )
        else:
            check(
                group.output_path.exists(),
                "quantized file",
                group.output_path.name,
            )
            check(
                not split_matches,
                "split outputs",
                f"{group.output_path.name} -> none expected",
            )
            check(
                not (group.output_path.parent / group.quant_label).exists(),
                "split output dir",
                group.quant_label,
            )

    assert not failures, "Integration checks failed:\n" + "\n".join(failures)
