"""
test_weekly_planner.py — Tests for the 7-day energy planner.
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

SRC_DIR = Path(__file__).parent.parent / "src"
DATA_DIR = Path(__file__).parent.parent / "data"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DATA_DIR))

from weekly_planner import build_weekly_plan


# ─────────────────────────────────────────────────────────────────────────────
# Plan generation
# ─────────────────────────────────────────────────────────────────────────────

def test_plan_generates_successfully(forecast_df, default_facilities):
    plan = build_weekly_plan(forecast_df, default_facilities)
    assert plan is not None
    assert len(plan.hourly_df) == 7 * 24
    assert len(plan.daily_df) == 7


def test_plan_responds_to_solar_scale(forecast_df, default_facilities):
    plan_hi = build_weekly_plan(forecast_df, default_facilities, solar_scale=1.5)
    plan_lo = build_weekly_plan(forecast_df, default_facilities, solar_scale=0.5)
    assert plan_hi.total_shortage_kwh <= plan_lo.total_shortage_kwh


def test_plan_responds_to_demand_scale(forecast_df, default_facilities):
    plan_lo = build_weekly_plan(forecast_df, default_facilities, demand_scale=0.8)
    plan_hi = build_weekly_plan(forecast_df, default_facilities, demand_scale=1.5)
    assert plan_hi.total_shortage_kwh >= plan_lo.total_shortage_kwh


def test_plan_grid_outage(forecast_df, default_facilities):
    """With grid_available=False, grid_energy should be 0."""
    plan = build_weekly_plan(
        forecast_df, default_facilities,
        grid_available=False, grid_capacity_kw=0.0,
    )
    assert plan.grid_energy_kwh < 0.1, (
        f"Grid energy {plan.grid_energy_kwh:.2f} kWh expected ~0 in outage scenario."
    )


def test_plan_storage_reduces_shortage(forecast_df, default_facilities):
    plan_no_stor = build_weekly_plan(
        forecast_df, default_facilities,
        storage_kwh=0.0, solar_scale=0.7,
    )
    plan_stor = build_weekly_plan(
        forecast_df, default_facilities,
        storage_kwh=100.0, solar_scale=0.7,
    )
    assert plan_stor.total_shortage_kwh <= plan_no_stor.total_shortage_kwh + 5.0


def test_plan_renewable_fraction_0_to_1(forecast_df, default_facilities):
    plan = build_weekly_plan(forecast_df, default_facilities)
    assert 0.0 <= plan.renewable_fraction <= 1.0 + 1e-6


def test_plan_surplus_non_negative(forecast_df, default_facilities):
    plan = build_weekly_plan(forecast_df, default_facilities)
    assert plan.total_surplus_kwh >= 0


def test_plan_shortage_non_negative(forecast_df, default_facilities):
    plan = build_weekly_plan(forecast_df, default_facilities)
    assert plan.total_shortage_kwh >= 0


def test_plan_shortage_appears_in_warnings(forecast_df, default_facilities):
    """Shortages must produce a warning message — not be silently suppressed."""
    plan = build_weekly_plan(
        forecast_df, default_facilities,
        solar_scale=0.1, grid_available=False, storage_kwh=0.0,
    )
    assert plan.total_shortage_kwh > 0
    real_warnings = [w for w in plan.shortage_warnings if w.strip()]
    assert len(real_warnings) > 0, "Shortages detected but no warning messages emitted."


def test_plan_flex_recommendations_not_empty(forecast_df, default_facilities):
    plan = build_weekly_plan(forecast_df, default_facilities)
    # At least one facility is flexible (water + school)
    assert len(plan.flex_recommendations) > 0


def test_plan_critical_reserve_positive(forecast_df, default_facilities):
    plan = build_weekly_plan(forecast_df, default_facilities)
    assert plan.critical_reserve_kwh_day > 0


def test_empty_forecast_raises(default_facilities):
    with pytest.raises(ValueError, match="empty"):
        build_weekly_plan(pd.DataFrame(), default_facilities)
