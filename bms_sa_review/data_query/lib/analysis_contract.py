"""Shared, read-only analysis contract for notebooks 02 and 03."""

from dataclasses import asdict, dataclass, field

import pandas as pd

from bms_sa_review.shared.ciccada_config import AS4777, SAI, TABLES


@dataclass(frozen=True)
class AnalysisConfig:
    database: str = SAI
    years: tuple = (2024, 2025)
    interval_h: float = AS4777["INTERVAL_H"]
    site_nonconf_threshold: float = AS4777["SITE_CONF_THRESH"]
    min_site_intervals: int = 1
    flex_selection: str = "exclude"
    rating_basis: str = "ac_capacity_kw"
    empirical_limit_basis: str = "s_99"
    voltage_aggregation: str = "avg"
    capability_profile: str = "review_corrected"
    day_night: str = "all"
    metadata_table: str = "meta_up23c"
    tables: dict = field(default_factory=lambda: dict(TABLES))

    def validate(self):
        if self.flex_selection not in {"exclude", "include", "only"}:
            raise ValueError("flex_selection must be exclude, include, or only")
        if self.rating_basis not in {"ac_capacity_kw", "s_99"}:
            raise ValueError("rating_basis must be ac_capacity_kw or s_99")
        if self.empirical_limit_basis not in {"ac_capacity_kw", "s_99"}:
            raise ValueError("empirical_limit_basis must be ac_capacity_kw or s_99")
        if self.voltage_aggregation not in {"avg", "max"}:
            raise ValueError("voltage_aggregation must be avg or max")
        if self.day_night not in {"all", "day", "night"}:
            raise ValueError("day_night must be all, day, or night")
        if not self.years:
            raise ValueError("years cannot be empty")
        return self


def manifest(config):
    """Return the methodological choices that must accompany every result."""
    config.validate()
    rows = [
        ("database", config.database),
        ("years", ", ".join(map(str, config.years))),
        ("interval_minutes", config.interval_h * 60),
        ("site_nonconf_threshold", config.site_nonconf_threshold),
        ("minimum_site_intervals", config.min_site_intervals),
        ("flex_selection", config.flex_selection),
        ("rating_basis", config.rating_basis),
        ("empirical_limit_basis", config.empirical_limit_basis),
        ("voltage_aggregation", config.voltage_aggregation),
        ("capability_profile", config.capability_profile),
        ("day_night", config.day_night),
    ]
    rows.extend((f"table.{key}", value) for key, value in config.tables.items())
    return pd.DataFrame(rows, columns=["setting", "value"])


def years_sql(years, column="year"):
    vals = ", ".join(str(int(y)) for y in years)
    return f"{column} IN ({vals})"


def day_night_sql(selection, column="day_night"):
    return "1 = 1" if selection == "all" else f"{column} = '{selection}'"


def flex_sql(selection, column="flex_export_detected"):
    if selection == "exclude":
        return f"coalesce({column}, False) = False"
    if selection == "only":
        return f"{column} = True"
    return "1 = 1"


def config_dict(config):
    out = asdict(config)
    out["years"] = list(config.years)
    return out
