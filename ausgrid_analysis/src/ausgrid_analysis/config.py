"""Typed configuration for the Ausgrid foundation pipeline."""

from __future__ import annotations

import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


@dataclass(frozen=True)
class PathConfig:
    telemetry_parquet: Path
    metadata_workbook: Path
    derived_root: Path

    @property
    def database_path(self) -> Path:
        return self.derived_root / "_duckdb" / "foundation.duckdb"

    @property
    def temp_directory(self) -> Path:
        return self.derived_root / "_duckdb" / "tmp"


@dataclass(frozen=True)
class MetadataConfig:
    sheet_name: str
    id_column: str


@dataclass(frozen=True)
class TelemetryConfig:
    site_column: str
    timestamp_column: str
    phase_column: str
    voltage_column: str
    current_column: str
    reactive_power_column: str
    active_power_column: str
    source_month_column: str
    source_file_column: str

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (
            self.site_column,
            self.timestamp_column,
            self.phase_column,
            self.voltage_column,
            self.current_column,
            self.reactive_power_column,
            self.active_power_column,
            self.source_month_column,
            self.source_file_column,
        )


@dataclass(frozen=True)
class AssumptionConfig:
    active_export_sign: int
    reactive_absorbing_sign: int
    source_timezone: str
    local_timezone: str
    power_sample_type: str
    measurement_location: str


@dataclass(frozen=True)
class QualityConfig:
    voltage_min_v: float
    voltage_max_v: float
    duplicate_float_tolerance: float


@dataclass(frozen=True)
class ProcessingConfig:
    site_bucket_count: int
    threads: int
    memory_limit: str
    parquet_compression: str


@dataclass(frozen=True)
class Delivery2Config:
    """Profiling defaults; all can be overridden in an optional [delivery2] table."""

    daytime_start_hour: int = 10
    daytime_end_hour: int = 14
    nighttime_start_hour: int = 0
    nighttime_end_hour: int = 4
    phase_mapping_min_signature_w: float = 100.0
    phase_mapping_high_margin_ratio: float = 0.35
    phase_mapping_medium_margin_ratio: float = 0.15


@dataclass(frozen=True)
class SourceScope:
    """A deterministic subset of the source parquet."""

    month: str | None = None
    site_bucket: int | None = None
    bucket_count: int = 32

    def validate(self) -> SourceScope:
        if self.month is not None and not _MONTH_RE.fullmatch(self.month):
            raise ValueError("month must use YYYY-MM format")
        if self.bucket_count <= 0:
            raise ValueError("bucket_count must be positive")
        if self.site_bucket is not None:
            if self.site_bucket < 0 or self.site_bucket >= self.bucket_count:
                raise ValueError(
                    f"site_bucket must be between 0 and {self.bucket_count - 1}"
                )
        return self

    @property
    def is_full(self) -> bool:
        return self.month is None and self.site_bucket is None

    @property
    def label(self) -> str:
        parts: list[str] = []
        if self.month is not None:
            parts.append(f"month_{self.month.replace('-', '_')}")
        if self.site_bucket is not None:
            parts.append(f"bucket_{self.site_bucket}_of_{self.bucket_count}")
        return "__".join(parts) if parts else "full"


@dataclass(frozen=True)
class FoundationConfig:
    paths: PathConfig
    metadata: MetadataConfig
    telemetry: TelemetryConfig
    assumptions: AssumptionConfig
    quality: QualityConfig
    processing: ProcessingConfig
    delivery2: Delivery2Config = field(default_factory=Delivery2Config)

    def validate(self, *, check_inputs: bool = False) -> FoundationConfig:
        for name, sign in (
            ("active_export_sign", self.assumptions.active_export_sign),
            ("reactive_absorbing_sign", self.assumptions.reactive_absorbing_sign),
        ):
            if sign not in (-1, 1):
                raise ValueError(f"{name} must be -1 or +1")

        if self.quality.voltage_min_v >= self.quality.voltage_max_v:
            raise ValueError("voltage_min_v must be below voltage_max_v")
        if self.quality.duplicate_float_tolerance <= 0:
            raise ValueError("duplicate_float_tolerance must be positive")
        if self.processing.site_bucket_count <= 0:
            raise ValueError("site_bucket_count must be positive")
        if self.processing.threads <= 0:
            raise ValueError("threads must be positive")
        if not self.processing.parquet_compression:
            raise ValueError("parquet_compression cannot be empty")
        d2 = self.delivery2
        for name in (
            "daytime_start_hour",
            "daytime_end_hour",
            "nighttime_start_hour",
            "nighttime_end_hour",
        ):
            value = getattr(d2, name)
            if value < 0 or value > 24:
                raise ValueError(f"{name} must be between 0 and 24")
        if d2.daytime_start_hour >= d2.daytime_end_hour:
            raise ValueError("daytime_start_hour must be before daytime_end_hour")
        if d2.nighttime_start_hour >= d2.nighttime_end_hour:
            raise ValueError("nighttime_start_hour must be before nighttime_end_hour")
        if not (
            0
            <= d2.phase_mapping_medium_margin_ratio
            <= d2.phase_mapping_high_margin_ratio
        ):
            raise ValueError("phase-mapping margin ratios are inconsistent")

        if check_inputs:
            missing = [
                path
                for path in (
                    self.paths.telemetry_parquet,
                    self.paths.metadata_workbook,
                )
                if not path.is_file()
            ]
            if missing:
                formatted = "\n".join(f"  - {path}" for path in missing)
                raise FileNotFoundError(f"Configured input files do not exist:\n{formatted}")
        return self

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["paths"] = {
            key: str(value) for key, value in result["paths"].items()
        }
        return result

    def scope(self, month: str | None, site_bucket: int | None) -> SourceScope:
        return SourceScope(
            month=month,
            site_bucket=site_bucket,
            bucket_count=self.processing.site_bucket_count,
        ).validate()


def _required_section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing [{name}] section in configuration")
    return section


def load_config(path: str | Path, *, check_inputs: bool = False) -> FoundationConfig:
    """Load and validate an analysis TOML file."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    paths = _required_section(data, "paths")
    metadata = _required_section(data, "metadata")
    telemetry = _required_section(data, "telemetry")
    assumptions = _required_section(data, "assumptions")
    quality = _required_section(data, "quality")
    processing = _required_section(data, "processing")
    delivery2 = data.get("delivery2", {})
    if not isinstance(delivery2, dict):
        raise ValueError("[delivery2] must be a TOML table")

    config = FoundationConfig(
        paths=PathConfig(
            telemetry_parquet=Path(paths["telemetry_parquet"]),
            metadata_workbook=Path(paths["metadata_workbook"]),
            derived_root=Path(paths["derived_root"]),
        ),
        metadata=MetadataConfig(**metadata),
        telemetry=TelemetryConfig(**telemetry),
        assumptions=AssumptionConfig(**assumptions),
        quality=QualityConfig(**quality),
        processing=ProcessingConfig(**processing),
        delivery2=Delivery2Config(**delivery2),
    )
    return config.validate(check_inputs=check_inputs)
