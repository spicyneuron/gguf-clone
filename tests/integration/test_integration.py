from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gguf import GGUFReader

from gguf_clone import config as config_mod
from gguf_clone.artifacts import Artifacts
from gguf_clone.main import run_pipeline
from gguf_clone.params import ParamsPayload, load_params
from gguf_clone.resolve import resolve_source_snapshot
from gguf_clone.split import split_needed


@dataclass(frozen=True)
class QuantOutput:
    quant_label: str
    params_path: Path
    output_path: Path
    payload: ParamsPayload


def _read_metadata(path: Path, keys: list[str]) -> dict[str, object | None]:
    reader = GGUFReader(str(path), "r")
    values: dict[str, object | None] = {}
    for key in keys:
        field = reader.get_field(key)
        values[key] = field.contents() if field is not None else None
    return values


def _split_outputs(output_path: Path, quant_label: str) -> list[Path]:
    split_dir = output_path.parent / quant_label
    if not split_dir.exists():
        return []
    return sorted(split_dir.glob(f"{output_path.stem}-*-of-*.gguf"))


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

    result = run_pipeline(config_path, overwrite_behavior="use")
    assert result == 0, "Main execution failed"

    config = config_mod.load_config(config_path)
    assert config is not None, "Failed to load config"
    assert config.extract_template is not None, "extract_template missing"
    assert config.quantize_gguf is not None, "quantize_gguf missing"

    ep = config.extract_template
    qg = config.quantize_gguf

    template_allow = list(ep.ggufs)
    if qg.imatrix:
        template_allow.append(qg.imatrix)
    if qg.copy_files:
        template_allow.extend(qg.copy_files)
    if qg.copy_metadata:
        template_allow.append("*.gguf")
    _ = resolve_source_snapshot(config.template, allow_patterns=template_allow)

    artifacts = Artifacts.from_config(config.output_dir, config.template, config.target)
    check(artifacts.quantized_gguf_dir.exists(), "quantized outputs directory exists")

    params_prefix = f"{artifacts.template_slug}-"
    params_suffix = "-gguf.json"
    params_files = sorted(
        p
        for p in artifacts.params_dir.iterdir()
        if p.name.startswith(params_prefix) and p.name.endswith(params_suffix)
    )
    check(bool(params_files), "params files discovered", str(artifacts.params_dir))

    outputs: list[QuantOutput] = []
    for params_path in params_files:
        quant_label = params_path.name[len(params_prefix) : -len(params_suffix)]
        payload = load_params(params_path)
        check(payload is not None, "params load", params_path.name)
        if payload is None:
            continue
        outputs.append(
            QuantOutput(
                quant_label=quant_label,
                params_path=params_path,
                output_path=artifacts.quantized_gguf(quant_label),
                payload=payload,
            )
        )

    check(
        len(outputs) == len(ep.ggufs),
        "quant count",
        f"expected={len(ep.ggufs)} actual={len(outputs)}",
    )

    staged_names: set[str] = set()
    for out in outputs:
        split_matches = _split_outputs(out.output_path, out.quant_label)
        output_path = out.output_path if out.output_path.exists() else None
        if output_path is None and split_matches:
            output_path = split_matches[0]

        if output_path is None:
            check(False, "quantized output", out.quant_label)
            continue

        if qg.copy_metadata:
            expected = {
                key: out.payload.template_metadata.get(key) for key in qg.copy_metadata
            }
            output_values = _read_metadata(output_path, qg.copy_metadata)
            for key in qg.copy_metadata:
                expected_value = expected.get(key)
                actual_value = output_values.get(key)
                ok = expected_value is not None and actual_value == expected_value
                if not ok:
                    detail = (
                        f"{output_path.name} {key} "
                        f"expected={_summarize_value(expected_value)} "
                        f"actual={_summarize_value(actual_value)}"
                    )
                    check(ok, "metadata match", detail)

        if qg.copy_files:
            staged_names.update(out.payload.staged_files)

        output_size = None
        if out.output_path.exists():
            output_size = out.output_path.stat().st_size
        elif split_matches:
            output_size = sum(path.stat().st_size for path in split_matches)

        if output_size is None:
            check(False, "quantized outputs", f"missing output for {out.quant_label}")
            continue

        needed = split_needed(output_size, qg.output_max_size)
        check(needed is not None, "split check", f"valid split size {qg.output_max_size}")
        if needed is None:
            continue

        if needed:
            check(
                bool(split_matches),
                "split outputs",
                f"{out.output_path.name} -> {len(split_matches)} file(s)",
            )
            check(
                (out.output_path.parent / out.quant_label).exists(),
                "split output dir",
                out.quant_label,
            )
            check(
                not out.output_path.exists(),
                "split source removed",
                out.output_path.name,
            )
        else:
            check(
                out.output_path.exists(),
                "quantized file",
                out.output_path.name,
            )
            check(
                not split_matches,
                "split outputs",
                f"{out.output_path.name} -> none expected",
            )
            check(
                not (out.output_path.parent / out.quant_label).exists(),
                "split output dir",
                out.quant_label,
            )

    if qg.copy_files:
        check(bool(staged_names), "staged files present")
        for name in sorted(staged_names):
            check(
                (artifacts.quantized_gguf_dir / name).exists(),
                "copied file",
                name,
            )
            check(
                (artifacts.template_files_dir / name).exists(),
                "staged file",
                name,
            )

    assert not failures, "Integration checks failed:\n" + "\n".join(failures)
