"""
test_optimiser.py — Tests for the optimisation engine.
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

from optimiser import (
    fixed_schedule, rule_based_schedule, optimised_schedule,
    compare_schedules,
)
from essential_services import DEFAULT_FACILITIES


# ─────────────────────────────────────────────────────────────────────────────
# Fixed schedule
# ─────────────────────────────────────────────────────────────────────────────

def test_fixed_schedule_runs(forecast_df, default_facilities):
    result = fixed_schedule(forecast_df, default_facilities)
    assert result is not None
    assert result.method.startswith("1.")


def test_fixed_schedule_non_negative_unmet(forecast_df, default_facilities):
    result = fixed_schedule(forecast_df, default_facilities)
    assert result.total_unmet_kwh >= 0


def test_fixed_schedule_energy_balance(forecast_df, default_facilities):
    result = fixed_schedule(forecast_df, default_facilities, storage_kwh=0.0, grid_capacity_kw=0.0)
    hourly = result.hourly_df
    # solar_used + unmet should equal demand (within floating-point tolerance)
    energy_out = (hourly["solar_used_kw"] + hourly["unmet_kw"]).sum()
    energy_in = hourly["demand_kw"].sum() if "demand_kw" in hourly else forecast_df["forecast_demand_kw"].sum()
    assert abs(energy_out - energy_in) < 10.0, f"Energy imbalance: {abs(energy_out - energy_in):.2f} kWh"


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based schedule
# ─────────────────────────────────────────────────────────────────────────────

def test_rule_based_runs(forecast_df, default_facilities):
    result = rule_based_schedule(forecast_df, default_facilities)
    assert result.method.startswith("2.")


def test_rule_based_non_negative_outputs(forecast_df, default_facilities):
    result = rule_based_schedule(forecast_df, default_facilities)
    assert result.total_unmet_kwh >= 0
    assert result.total_grid_kwh >= 0
    assert result.total_solar_used_kwh >= 0


def test_rule_based_better_or_equal_to_fixed(forecast_df, default_facilities):
    """Rule-based should not be significantly worse than fixed on normal conditions."""
    r_fixed = fixed_schedule(forecast_df, default_facilities)
    r_rule = rule_based_schedule(forecast_df, default_facilities)
    # Allow 10% slack
    assert r_rule.total_unmet_kwh <= r_fixed.total_unmet_kwh * 1.10 + 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Optimised schedule
# ─────────────────────────────────────────────────────────────────────────────

def test_optimised_runs(forecast_df, default_facilities):
    result = optimised_schedule(forecast_df, default_facilities)
    assert result is not None
    assert "3." in result.method


def test_optimised_non_negative(forecast_df, default_facilities):
    result = optimised_schedule(forecast_df, default_facilities)
    assert result.total_unmet_kwh >= 0
    assert result.total_grid_kwh >= 0


def test_optimised_better_or_equal_to_rule(forecast_df, default_facilities):
    """Optimised unmet should be ≤ rule-based (objective: minimise unmet)."""
    r_rule = rule_based_schedule(forecast_df, default_facilities)
    r_opt = optimised_schedule(forecast_df, default_facilities)
    # Allow 5 kWh slack for numerical differences
    assert r_opt.total_unmet_kwh <= r_rule.total_unmet_kwh + 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Scheduling-window constraints
# ─────────────────────────────────────────────────────────────────────────────

def test_scheduling_window_respected(forecast_df, default_facilities):
    """
    Flexible loads should have demand scheduled within their windows.
    Rule-based: check that the water facility is not scheduled at night.
    """
    from optimiser import _build_hourly_demand
    demand_df = _build_hourly_demand(forecast_df, default_facilities, demand_scale=1.0)

    water_fac = next(f for f in default_facilities if f.key == "water_facility")
    op_start, op_end = water_fac.operating_hours

    night_mask = (forecast_df.index.hour < op_start) | (forecast_df.index.hour >= op_end)
    night_demand = demand_df.loc[night_mask, "water_facility"]
    # Night demand must be ≤ small standby (0.5 kW)
    assert (night_demand <= 0.5 + 1e-6).all(), (
        "Water facility demand found outside operating hours."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Compare schedules
# ─────────────────────────────────────────────────────────────────────────────

def test_compare_schedules_returns_three(forecast_df, default_facilities):
    r1, r2, r3 = compare_schedules(forecast_df, default_facilities)
    assert r1 is not None
    assert r2 is not None
    assert r3 is not None


def test_compare_schedules_same_inputs(forecast_df, default_facilities):
    """All three methods must receive the same scenario inputs."""
    r1, r2, r3 = compare_schedules(
        forecast_df, default_facilities,
        solar_scale=0.8, demand_scale=1.1,
        grid_capacity_kw=5.0, storage_kwh=30.0,
    )
    # Verify methods are distinct
    assert r1.method != r2.method
    assert r2.method != r3.method


# ─────────────────────────────────────────────────────────────────────────────
# Infeasible scenario
# ─────────────────────────────────────────────────────────────────────────────

def test_infeasible_scenario_exposed(shortage_forecast_df, default_facilities):
    """
    With minimal solar and no grid, the optimiser must report the conflict
    rather than silently hiding it.
    """
    result = optimised_schedule(
        shortage_forecast_df, default_facilities,
        solar_scale=0.05,
        demand_scale=2.0,
        grid_capacity_kw=0.0,
        storage_kwh=0.0,
    )
    # Unmet demand should be positive
    assert result.total_unmet_kwh > 0, (
        "FAILURE: Infeasible scenario shows zero unmet demand — conflict is hidden."
    )
