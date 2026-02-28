from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

import yaml
from pydantic import BaseModel, BeforeValidator, ValidationError


def _ensure_list(v: str | list[str]) -> list[str]:
    return [v] if isinstance(v, str) else v


StrList = Annotated[list[str], BeforeValidator(_ensure_list)]


class SourceConfig(BaseModel):
    template: str
    target: str


class ExtractTemplateConfig(BaseModel):
    ggufs: StrList
    targets: list[Literal["gguf", "mlx"]] = ["gguf"]
    mlx_arch: str | None = None


class QuantizeGgufConfig(BaseModel):
    target_gguf: str | None = None
    target_convert: bool = True
    imatrix: str | None = "*imatrix*"
    output_max_size: str = "50G"
    copy_metadata: StrList = []
    copy_files: StrList = []
    apply_metadata: dict[str, str] = {
        "general.quantized_by": "https://github.com/spicyneuron/gguf-clone"
    }


class QuantizeMlxConfig(BaseModel):
    group_size: int = 64
    trust_remote_code: bool = False


class ConfigFile(BaseModel):
    version: int = 2
    source: SourceConfig
    output_dir: str = "output"
    extract_template: ExtractTemplateConfig | None = None
    quantize_gguf: QuantizeGgufConfig | None = None
    quantize_mlx: QuantizeMlxConfig | None = None


@dataclass(frozen=True)
class SourceRef:
    """Classified reference to a model source -- no I/O performed."""

    repo: str | None
    path: Path | None


@dataclass(frozen=True)
class RunConfig:
    template: SourceRef
    target: SourceRef
    output_dir: Path
    extract_template: ExtractTemplateConfig | None
    quantize_gguf: QuantizeGgufConfig | None
    quantize_mlx: QuantizeMlxConfig | None


def _resolve_source(value: str, config_dir: Path) -> SourceRef:
    """Resolve a source string to either a local directory or HF repo.

    If the value resolves to an existing directory (absolute or relative to
    config_dir), treat it as local. Otherwise assume HF repo ID.
    """
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        if candidate.is_dir():
            return SourceRef(repo=None, path=candidate)
    else:
        resolved = (config_dir / candidate).resolve()
        if resolved.is_dir():
            return SourceRef(repo=None, path=resolved)
    return SourceRef(repo=value, path=None)


def _resolve_template(value: str, config_dir: Path) -> SourceRef | str:
    """Resolve template source with GGUF validation for local dirs.

    Returns a SourceRef on success or an error string on failure.
    """
    ref = _resolve_source(value, config_dir)
    if ref.path is not None:
        ggufs = list(ref.path.rglob("*.gguf"))
        if not ggufs:
            return f"Template directory contains no .gguf files: {ref.path}"
    return ref


def load_config(path: Path) -> RunConfig | None:
    try:
        text = path.read_text()
    except FileNotFoundError:
        print(f"Config file not found: {path}")
        return None
    except OSError as exc:
        print(f"Failed to read config: {exc}")
        return None

    try:
        data = cast(object, yaml.safe_load(text))
        config = ConfigFile.model_validate(data)
    except (yaml.YAMLError, ValidationError) as exc:
        print(f"Failed to read config: {exc}")
        return None

    if config.version != 2:
        print(f"Unsupported config version: {config.version}")
        return None

    config_dir = path.parent.resolve()
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = (config_dir / output_dir).resolve()

    template_result = _resolve_template(config.source.template, config_dir)
    if isinstance(template_result, str):
        print(template_result)
        return None

    return RunConfig(
        template=template_result,
        target=_resolve_source(config.source.target, config_dir),
        output_dir=output_dir,
        extract_template=config.extract_template,
        quantize_gguf=config.quantize_gguf,
        quantize_mlx=config.quantize_mlx,
    )
