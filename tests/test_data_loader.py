"""
test_data_loader.py — Tests for data loading and validation.
"""
import sys
from pathlib import Path
import pytest
import pandas as pd

SRC_DIR = Path(__file__).parent.parent / "src"
DATA_DIR = Path(__file__).parent.parent / "data"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DATA_DIR))

from data_loader import load_data, get_facility_columns, REQUIRED_COLUMNS
from validation import validate_dataframe, validate_facility_config, validate_scenario_params


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def test_load_data_returns_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_data_has_required_columns():
    df = load_data()
    for col in REQUIRED_COLUMNS:
        if col != "timestamp":  # timestamp becomes the index
            assert col in df.columns, f"Missing column: {col}"


def test_load_data_datetime_index():
    df = load_data()
    assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex"


def test_load_data_no_negative_power():
    df = load_data()
    for col in ["solar_generation_kw", "total_demand_kw"]:
        assert (df[col] >= -0.01).all(), f"Negative values in {col}"


def test_load_data_sorted():
    df = load_data()
    assert df.index.is_monotonic_increasing, "Timestamps not sorted"


def test_get_facility_columns():
    cols = get_facility_columns()
    assert "health_centre" in cols
    assert "water_facility" in cols
    assert "school" in cols


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame validation
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_good_dataframe(raw_df):
    result = validate_dataframe(raw_df)
    assert result.is_valid, f"Validation errors: {result.errors}"


def test_validate_empty_dataframe():
    result = validate_dataframe(pd.DataFrame())
    assert not result.is_valid
    assert any("empty" in e.lower() for e in result.errors)


def test_validate_missing_data():
    """Invalid data: negative power values should flag an error."""
    df = load_data().copy()
    df["solar_generation_kw"] = -1.0
    result = validate_dataframe(df)
    assert not result.is_valid
    assert any("negative" in e.lower() for e in result.errors)


# ─────────────────────────────────────────────────────────────────────────────
# Facility config validation
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_facility_good():
    config = {
        "name": "Test",
        "priority": "critical",
        "min_energy_kwh_day": 20.0,
        "operating_hours": (6, 20),
    }
    result = validate_facility_config(config)
    assert result.is_valid


def test_validate_facility_bad_priority():
    config = {
        "name": "Test",
        "priority": "urgent",   # invalid
        "min_energy_kwh_day": 10.0,
        "operating_hours": (0, 24),
    }
    result = validate_facility_config(config)
    assert not result.is_valid


def test_validate_facility_bad_hours():
    config = {
        "name": "Test",
        "priority": "important",
        "min_energy_kwh_day": 5.0,
        "operating_hours": (20, 6),  # end < start → invalid
    }
    result = validate_facility_config(config)
    assert not result.is_valid


# ─────────────────────────────────────────────────────────────────────────────
# Scenario params validation
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_scenario_params_good():
    params = {
        "solar_scale": 0.7,
        "demand_scale": 1.2,
        "storage_kwh": 50.0,
        "grid_available": True,
    }
    result = validate_scenario_params(params)
    assert result.is_valid


def test_validate_scenario_params_bad_solar():
    result = validate_scenario_params({"solar_scale": -0.5, "demand_scale": 1.0,
                                       "storage_kwh": 0.0, "grid_available": True})
    assert not result.is_valid


def test_validate_scenario_params_negative_storage():
    result = validate_scenario_params({"solar_scale": 1.0, "demand_scale": 1.0,
                                       "storage_kwh": -10.0, "grid_available": True})
    assert not result.is_valid
