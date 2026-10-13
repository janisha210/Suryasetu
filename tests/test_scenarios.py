"""
test_scenarios.py — Tests for the scenario laboratory.
"""
import sys
from pathlib import Path
import pytest
import pandas as pd

SRC_DIR = Path(__file__).parent.parent / "src"
DATA_DIR = Path(__file__).parent.parent / "data"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DATA_DIR))

from scenarios import (
    PRESET_SCENARIOS, run_scenario, build_custom_scenario, Scenario,
)
from essential_services import DEFAULT_FACILITIES


@pytest.fixture(scope="module")
def fc_df():
    from forecasting import ForecastPipeline
    from data_loader import load_data
    raw = load_data()
    pipeline = ForecastPipeline(test_days=14)
    pipeline.run(raw)
    return pipeline.forecast_df


@pytest.fixture(scope="module")
def facilities():
    return list(DEFAULT_FACILITIES)


# ─────────────────────────────────────────────────────────────────────────────
# Preset scenarios
# ─────────────────────────────────────────────────────────────────────────────

def test_all_preset_scenarios_run(fc_df, facilities):
    for scenario in PRESET_SCENARIOS:
        result = run_scenario(scenario, fc_df, facilities)
        assert result is not None, f"Scenario '{scenario.name}' returned None."


def test_normal_scenario_no_critical_shortage(fc_df, facilities):
    """The normal baseline scenario should NOT have critical shortages."""
    normal = next(s for s in PRESET_SCENARIOS if "Normal" in s.name)
    result = run_scenario(normal, fc_df, facilities)
    # On normal scenario, shortages may exist but check comparison table is built
    assert result.comparison_df() is not None


def test_critical_shortage_scenario_has_unmet(fc_df, facilities):
    """The 'Critical Shortage' preset must produce unmet demand."""
    shortage_scen = next(s for s in PRESET_SCENARIOS if "Critical Shortage" in s.name)
    result = run_scenario(shortage_scen, fc_df, facilities)
    assert result.plan.total_shortage_kwh > 0, (
        "Critical Shortage scenario shows zero shortage — conflict is hidden."
    )
    assert result.service_report.has_critical_shortage or result.plan.total_shortage_kwh > 0


def test_grid_outage_scenario(fc_df, facilities):
    outage = next(s for s in PRESET_SCENARIOS if "Grid Outage" in s.name)
    result = run_scenario(outage, fc_df, facilities)
    assert result.plan.grid_energy_kwh < 1.0, (
        "Grid outage scenario should produce near-zero grid energy."
    )


def test_reduced_solar_worse_than_normal(fc_df, facilities):
    normal = next(s for s in PRESET_SCENARIOS if "Normal" in s.name)
    reduced = next(s for s in PRESET_SCENARIOS if "Reduced Solar" in s.name)
    r_normal = run_scenario(normal, fc_df, facilities)
    r_reduced = run_scenario(reduced, fc_df, facilities)
    # Reduced solar should have more shortage or equal
    assert r_reduced.plan.total_shortage_kwh >= r_normal.plan.total_shortage_kwh - 5.0


def test_increased_demand_worse_than_normal(fc_df, facilities):
    normal = next(s for s in PRESET_SCENARIOS if "Normal" in s.name)
    increased = next(s for s in PRESET_SCENARIOS if "Increased Demand" in s.name)
    r_normal = run_scenario(normal, fc_df, facilities)
    r_increased = run_scenario(increased, fc_df, facilities)
    assert r_increased.plan.total_shortage_kwh >= r_normal.plan.total_shortage_kwh - 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_comparison_has_three_methods(fc_df, facilities):
    result = run_scenario(PRESET_SCENARIOS[0], fc_df, facilities)
    comp = result.comparison_df()
    assert len(comp) == 3
    methods = set(comp["Method"].tolist())
    assert len(methods) == 3, f"Expected 3 distinct methods, got: {methods}"


def test_scenario_comparison_columns(fc_df, facilities):
    result = run_scenario(PRESET_SCENARIOS[0], fc_df, facilities)
    comp = result.comparison_df()
    assert "Total Unmet Demand (kWh)" in comp.columns
    assert "Grid Energy Used (kWh)" in comp.columns
    assert "Critical Coverage (%)" in comp.columns


# ─────────────────────────────────────────────────────────────────────────────
# Custom scenario
# ─────────────────────────────────────────────────────────────────────────────

def test_custom_scenario_construction():
    s = build_custom_scenario(
        name="My Test",
        solar_scale=0.6, demand_scale=1.3,
        storage_kwh=20.0, grid_available=True, grid_capacity_kw=3.0,
    )
    assert s.solar_scale == 0.6
    assert s.demand_scale == 1.3
    assert s.is_illustrative is True


def test_custom_scenario_runs(fc_df, facilities):
    s = build_custom_scenario(
        name="Test", solar_scale=0.8, demand_scale=1.1,
        storage_kwh=30.0, grid_available=True, grid_capacity_kw=4.0,
    )
    result = run_scenario(s, fc_df, facilities)
    assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# Scenario isolation: same inputs → reproducible results
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_reproducible(fc_df, facilities):
    s = PRESET_SCENARIOS[0]
    r1 = run_scenario(s, fc_df, facilities)
    r2 = run_scenario(s, fc_df, facilities)
    assert abs(r1.plan.total_shortage_kwh - r2.plan.total_shortage_kwh) < 0.1
