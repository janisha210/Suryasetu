"""
data_loader.py — Load and validate the synthetic dataset.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
CSV_PATH = DATA_DIR / "synthetic_energy_data.csv"

REQUIRED_COLUMNS = [
    "timestamp",
    "hour",
    "day_of_week",
    "day_of_year",
    "cloud_factor",
    "solar_irradiance_factor",
    "solar_generation_kw",
    "health_centre_demand_kw",
    "water_facility_demand_kw",
    "school_demand_kw",
    "total_demand_kw",
    "net_balance_kw",
]

NUMERIC_COLUMNS = [
    "hour",
    "day_of_week",
    "day_of_year",
    "cloud_factor",
    "solar_irradiance_factor",
    "solar_generation_kw",
    "health_centre_demand_kw",
    "water_facility_demand_kw",
    "school_demand_kw",
    "total_demand_kw",
    "net_balance_kw",
]


def _generate_if_missing() -> None:
    """Run the generator script if the CSV does not exist yet."""
    if not CSV_PATH.exists():
        gen = DATA_DIR / "synthetic_data_generator.py"
        if not gen.exists():
            raise FileNotFoundError(
                f"Neither the dataset ({CSV_PATH}) nor the generator ({gen}) "
                "could be found. Please check your installation."
            )
        subprocess.run([sys.executable, str(gen)], check=True)


def load_data(path: Path | None = None, force_regenerate: bool = False) -> pd.DataFrame:
    """Load the synthetic energy dataset.

    Parameters
    ----------
    path:
        Optional path override (useful in tests).
    force_regenerate:
        If True, re-run the generator even if the CSV exists.

    Returns
    -------
    pd.DataFrame with a DatetimeIndex (UTC-naïve).
    """
    target = path or CSV_PATH

    if force_regenerate or not target.exists():
        _generate_if_missing()

    df = pd.read_csv(target, parse_dates=["timestamp"])

    # Basic column presence check
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            "Re-generate the dataset with data/synthetic_data_generator.py."
        )

    # Sort and index
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")

    # Coerce numeric columns; warn on unexpected NaNs
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n_nan = df[NUMERIC_COLUMNS].isna().sum().sum()
    if n_nan > 0:
        # Lag columns naturally have NaNs at the head; tolerate them
        lag_cols = [c for c in df.columns if "lag" in c]
        non_lag_nan = df[[c for c in NUMERIC_COLUMNS if c not in lag_cols]].isna().sum().sum()
        if non_lag_nan > 0:
            raise ValueError(
                f"Dataset contains {non_lag_nan} unexpected NaN values in "
                "non-lag columns. Check the source CSV for data-quality issues."
            )

    return df


def get_last_n_days(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return the last *n* calendar days from the dataframe."""
    cutoff = df.index.max() - pd.Timedelta(days=n)
    return df[df.index > cutoff].copy()


def get_facility_columns() -> dict[str, str]:
    """Return a mapping of facility key → display label."""
    return {
        "health_centre": "Primary Health Centre",
        "water_facility": "Drinking-Water Facility",
        "school": "Community School",
    }
