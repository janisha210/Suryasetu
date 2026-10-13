"""
validation.py — Input and data validation helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Run schema and value checks on the loaded dataset."""
    result = ValidationResult(is_valid=True)

    if df is None or len(df) == 0:
        result.add_error("Dataset is empty or could not be loaded.")
        return result

    # Timestamp index
    if not isinstance(df.index, pd.DatetimeIndex):
        result.add_error("Dataset index must be a DatetimeIndex.")

    # Duplicate timestamps
    dupes = df.index.duplicated().sum()
    if dupes > 0:
        result.add_error(f"Dataset contains {dupes} duplicate timestamps.")

    # Monotonic timestamps
    if not df.index.is_monotonic_increasing:
        result.add_warning("Timestamps are not strictly monotonically increasing.")

    # Non-negative power values
    for col in ["solar_generation_kw", "total_demand_kw",
                "health_centre_demand_kw", "water_facility_demand_kw",
                "school_demand_kw"]:
        if col in df.columns:
            n_neg = (df[col] < -0.01).sum()
            if n_neg > 0:
                result.add_error(
                    f"Column '{col}' contains {n_neg} negative values "
                    "(expected non-negative power readings)."
                )

    # Hour validity
    if "hour" in df.columns:
        invalid_hours = (~df["hour"].between(0, 23)).sum()
        if invalid_hours:
            result.add_error(f"'hour' column has {invalid_hours} values outside 0–23.")

    # Cloud factor 0–1
    if "cloud_factor" in df.columns:
        out_of_range = (~df["cloud_factor"].between(0, 1)).sum()
        if out_of_range:
            result.add_warning(
                f"'cloud_factor' has {out_of_range} values outside [0, 1]."
            )

    # Minimum row count
    if len(df) < 24 * 14:
        result.add_warning(
            f"Dataset has only {len(df)} rows. "
            "At least 14 days (336 rows) recommended for reliable forecasts."
        )

    return result


def validate_scenario_params(params: dict[str, Any]) -> ValidationResult:
    """Validate scenario editor parameters."""
    result = ValidationResult(is_valid=True)

    solar_scale = params.get("solar_scale", 1.0)
    demand_scale = params.get("demand_scale", 1.0)
    storage_kwh = params.get("storage_kwh", 0.0)
    grid_available = params.get("grid_available", True)

    if not 0.01 <= solar_scale <= 5.0:
        result.add_error(
            f"solar_scale={solar_scale} is outside the valid range [0.01, 5.0]."
        )

    if not 0.01 <= demand_scale <= 5.0:
        result.add_error(
            f"demand_scale={demand_scale} is outside the valid range [0.01, 5.0]."
        )

    if storage_kwh < 0:
        result.add_error(f"storage_kwh={storage_kwh} must be ≥ 0.")

    if storage_kwh > 10_000:
        result.add_warning(
            f"storage_kwh={storage_kwh} is unusually large for a community system."
        )

    if not isinstance(grid_available, bool):
        result.add_error("'grid_available' must be a boolean (True/False).")

    return result


def validate_facility_config(config: dict[str, Any]) -> ValidationResult:
    """Validate a single facility configuration dictionary."""
    result = ValidationResult(is_valid=True)

    name = config.get("name", "<unnamed>")

    priority = config.get("priority", "")
    if priority not in ("critical", "important", "flexible"):
        result.add_error(
            f"Facility '{name}': priority must be 'critical', 'important', or 'flexible'."
        )

    min_e = config.get("min_energy_kwh_day", 0)
    if min_e < 0:
        result.add_error(f"Facility '{name}': min_energy_kwh_day must be ≥ 0.")

    op = config.get("operating_hours", (0, 24))
    if not (0 <= op[0] < op[1] <= 24):
        result.add_error(
            f"Facility '{name}': operating_hours {op} is invalid. "
            "Must satisfy 0 ≤ start < end ≤ 24."
        )

    return result
