"""Centralized output path conventions.

Every stage derives paths from a single Artifacts instance so naming
and directory layout stay consistent across the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import SourceRef


def source_slug(ref: SourceRef) -> str:
    if ref.repo:
        return ref.repo.replace("/", "-")
    if not ref.path:
        return "model"
    name = ref.path.stem if ref.path.suffix else ref.path.name
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    return slug or "model"


def source_desc(ref: SourceRef) -> str:
    if ref.repo:
        return f"repo:{ref.repo}"
    if ref.path:
        return f"path:{ref.path}"
    return "unknown"


def quant_label_from_stem(stem: str) -> str | None:
    ud = re.search(r"(UD-[A-Z0-9_]+)", stem, re.IGNORECASE)
    if ud:
        return ud.group(1).upper()

    patterns = [
        re.compile(r"(IQ\d+_[A-Z0-9_]+)", re.IGNORECASE),
        re.compile(r"(Q\d+_[A-Z0-9_]+)", re.IGNORECASE),
        re.compile(r"(BF16|F16|F32)", re.IGNORECASE),
    ]
    for pattern in patterns:
        matches: list[str] = pattern.findall(stem)
        if matches:
            return matches[-1].upper()
    return None


@dataclass(frozen=True)
class Artifacts:
    output_dir: Path
    template_slug: str
    target_slug: str

    @staticmethod
    def from_config(
        output_dir: Path, template: SourceRef, target: SourceRef
    ) -> "Artifacts":
        return Artifacts(
            output_dir=output_dir,
            template_slug=source_slug(template),
            target_slug=source_slug(target),
        )

    # -- directory roots --

    @property
    def converted_dir(self) -> Path:
        return self.output_dir / "converted"

    @property
    def params_dir(self) -> Path:
        return self.output_dir / "params"

    @property
    def template_files_dir(self) -> Path:
        return self.output_dir / "template_files"

    @property
    def quantized_gguf_dir(self) -> Path:
        return self.output_dir / "quantized" / "gguf"

    @property
    def quantized_mlx_dir(self) -> Path:
        return self.output_dir / "quantized" / "mlx"

    # -- specific file paths --

    def converted_gguf(self) -> Path:
        return self.converted_dir / f"{self.target_slug}.gguf"

    def params_gguf(self, quant_label: str) -> Path:
        return self.params_dir / f"{self.template_slug}-{quant_label}-gguf.json"

    def params_mlx(self, quant_label: str) -> Path:
        return self.params_dir / f"{self.template_slug}-{quant_label}-mlx.json"

    def quantized_gguf(self, quant_label: str) -> Path:
        return self.quantized_gguf_dir / f"{self.target_slug}-{quant_label}.gguf"

    def quantized_mlx(self, quant_label: str) -> Path:
        return self.quantized_mlx_dir / quant_label

    def mkdir_all(self) -> None:
        """Create all output directories (idempotent)."""
        for d in (
            self.converted_dir,
            self.params_dir,
            self.template_files_dir,
            self.quantized_gguf_dir,
            self.quantized_mlx_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
