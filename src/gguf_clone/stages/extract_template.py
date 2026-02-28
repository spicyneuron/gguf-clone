from __future__ import annotations

from pathlib import Path

from .. import config as config_mod
from ..artifacts import Artifacts, quant_label_from_stem
from ..common import confirm_overwrite, log_stage, log_success, remove_files
from ..files import copy_template_files, match_copy_files
from ..metadata import extract_template_metadata
from ..params import ParamsPayload, build_params, copy_imatrix, save_params_payload
from ..resolve import ModelResolutionError, match_pattern, resolve_source_snapshot


def _resolve_template_snapshot(
    ref: config_mod.SourceRef,
    allow_patterns: list[str],
) -> Path | None:
    try:
        return resolve_source_snapshot(ref, allow_patterns=allow_patterns or None)
    except ModelResolutionError as exc:
        print(str(exc))
        return None


def _template_allow_patterns(config: config_mod.RunConfig) -> list[str]:
    ep = config.extract_template
    if ep is None:
        return []

    allow_patterns = list(ep.ggufs)
    qg = config.quantize_gguf
    if qg is None:
        return allow_patterns

    if qg.imatrix:
        allow_patterns.append(qg.imatrix)
    if qg.copy_files:
        allow_patterns.extend(qg.copy_files)
    if qg.copy_metadata:
        allow_patterns.append("*.gguf")
    return allow_patterns


def run_extract_template_stage(
    config: config_mod.RunConfig,
    artifacts: Artifacts,
) -> int:
    ep = config.extract_template
    if ep is None:
        print("extract_template section missing from config.")
        return 1

    artifacts.mkdir_all()

    template_snapshot = _resolve_template_snapshot(
        config.template,
        _template_allow_patterns(config),
    )
    if template_snapshot is None:
        return 1

    gguf_candidates = sorted(
        path for path in template_snapshot.rglob("*.gguf") if path.is_file()
    )
    if not gguf_candidates:
        print(f"No .gguf files found in template: {template_snapshot}")
        return 1

    qg = config.quantize_gguf
    imatrix_rel = ""
    if qg is not None and qg.imatrix:
        imatrix_candidates = [
            path for path in template_snapshot.rglob("*") if path.is_file()
        ]
        imatrix_matches = match_pattern(
            template_snapshot,
            imatrix_candidates,
            qg.imatrix,
            "imatrix",
            require_single=True,
        )
        if not imatrix_matches:
            return 1
        imatrix_rel = copy_imatrix(
            imatrix_matches[0],
            artifacts.params_dir,
            prefix="params",
        )

    staged_file_names: list[str] = []
    if qg is not None and qg.copy_files:
        matched_files = match_copy_files(template_snapshot, qg.copy_files)
        if matched_files is None:
            return 1
        staged_file_names = sorted({path.name for path in matched_files})

        stage_indent = log_stage("Staging template files for later stages")
        result = copy_template_files(
            template_snapshot,
            qg.copy_files,
            artifacts.template_files_dir,
            indent=stage_indent,
        )
        if result != 0:
            return result

    for gguf_pattern in ep.ggufs:
        template_group = match_pattern(
            template_snapshot,
            gguf_candidates,
            gguf_pattern,
            "template GGUF",
        )
        if not template_group:
            return 1

        stem_labels = {
            label for path in template_group if (label := quant_label_from_stem(path.stem))
        }
        if len(stem_labels) > 1:
            print("Template GGUF pattern matched multiple quant labels:")
            for label in sorted(stem_labels):
                print(f"  {label}")
            return 1

        try:
            params = build_params(template_group)
        except ValueError as exc:
            print(str(exc))
            return 1

        quant_label = next(iter(stem_labels), None) or params.default_type
        template_gguf = template_group[0].relative_to(template_snapshot).as_posix()

        template_metadata: dict[str, str] = {}
        if qg is not None and qg.copy_metadata:
            extracted = extract_template_metadata(
                template_group[0],
                qg.copy_metadata,
            )
            if extracted is None:
                return 1
            template_metadata = extracted

        gguf_path = artifacts.params_gguf(quant_label)
        action = confirm_overwrite([gguf_path], "params", indent="")
        if action == "cancel":
            return 1
        if action == "use":
            log_success(f"Using existing params: {gguf_path}")
        else:
            if gguf_path.exists() and not remove_files([gguf_path]):
                return 1
            payload = ParamsPayload(
                tensor_types=params.tensor_types,
                default_type=params.default_type,
                imatrix=imatrix_rel,
                template_metadata=template_metadata,
                staged_files=staged_file_names,
                template_gguf=template_gguf,
            )
            save_params_payload(payload, gguf_path)
            log_success(f"Params saved to {gguf_path}")

    return 0
