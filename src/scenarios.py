"""
scenarios.py — Scenario laboratory definitions and runner (Module E).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from essential_services import Facility, evaluate_service_coverage, ServiceProtectionReport
from optimiser import compare_schedules, ScheduleResult
from weekly_planner import build_weekly_plan, WeeklyPlan


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    description: str
    solar_scale: float = 1.0
    demand_scale: float = 1.0
    storage_kwh: float = 50.0
    grid_available: bool = True
    grid_capacity_kw: float = 5.0
    is_illustrative: bool = True      # always True for this prototype
    notes: str = ""


PRESET_SCENARIOS: list[Scenario] = [
    Scenario(
        name="Normal (Illustrative Baseline)",
        description="Typical operating conditions with normal solar and demand.",
        solar_scale=1.0,
        demand_scale=1.0,
        storage_kwh=50.0,
        grid_available=True,
        grid_capacity_kw=5.0,
        notes="All values are illustrative and synthetic.",
    ),
    Scenario(
        name="Reduced Solar (−30%)",
        description="Solar generation reduced by 30% (e.g., heavy cloud or seasonal low).",
        solar_scale=0.70,
        demand_scale=1.0,
        storage_kwh=50.0,
        grid_available=True,
        grid_capacity_kw=5.0,
        notes="Illustrative scenario: solar_scale = 0.70.",
    ),
    Scenario(
        name="Increased Demand (+20%)",
        description="Community electricity demand 20% above normal.",
        solar_scale=1.0,
        demand_scale=1.20,
        storage_kwh=50.0,
        grid_available=True,
        grid_capacity_kw=5.0,
        notes="Illustrative scenario: demand_scale = 1.20.",
    ),
    Scenario(
        name="Reduced Storage",
        description="Battery storage capacity cut to 10 kWh.",
        solar_scale=1.0,
        demand_scale=1.0,
        storage_kwh=10.0,
        grid_available=True,
        grid_capacity_kw=5.0,
        notes="Storage reduced to 10 kWh from normal 50 kWh.",
    ),
    Scenario(
        name="Grid Outage",
        description="No grid supply available; rely only on solar and storage.",
        solar_scale=1.0,
        demand_scale=1.0,
        storage_kwh=50.0,
        grid_available=False,
        grid_capacity_kw=0.0,
        notes="Grid unavailable. All energy must come from solar + storage.",
    ),
    Scenario(
        name="Critical Shortage",
        description="Critical demand far exceeds supply (solar −60%, demand +30%).",
        solar_scale=0.40,
        demand_scale=1.30,
        storage_kwh=10.0,
        grid_available=False,
        grid_capacity_kw=0.0,
        notes=(
            "Shortage scenario: solar_scale=0.40, demand_scale=1.30, "
            "no grid, minimal storage. Expect unmet critical demand."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario: Scenario
    plan: WeeklyPlan
    service_report: ServiceProtectionReport
    schedule_fixed: ScheduleResult
    schedule_rule: ScheduleResult
    schedule_opt: ScheduleResult

    def comparison_df(self) -> pd.DataFrame:
        rows = [
            self.schedule_fixed.summary_dict(),
            self.schedule_rule.summary_dict(),
            self.schedule_opt.summary_dict(),
        ]
        return pd.DataFrame(rows)


def run_scenario(
    scenario: Scenario,
    forecast_df: pd.DataFrame,
    facilities: list[Facility],
) -> ScenarioResult:
    """Run a full scenario: plan + service coverage + three schedule methods."""

    plan = build_weekly_plan(
        forecast_df=forecast_df,
        facilities=facilities,
        solar_scale=scenario.solar_scale,
        demand_scale=scenario.demand_scale,
        grid_capacity_kw=scenario.grid_capacity_kw,
        storage_kwh=scenario.storage_kwh,
        grid_available=scenario.grid_available,
    )

    service_report = evaluate_service_coverage(
        forecast_df=forecast_df,
        facilities=facilities,
        solar_scale=scenario.solar_scale,
        demand_scale=scenario.demand_scale,
    )

    r1, r2, r3 = compare_schedules(
        forecast_df=forecast_df,
        facilities=facilities,
        solar_scale=scenario.solar_scale,
        demand_scale=scenario.demand_scale,
        grid_capacity_kw=scenario.grid_capacity_kw if scenario.grid_available else 0.0,
        storage_kwh=scenario.storage_kwh,
    )

    return ScenarioResult(
        scenario=scenario,
        plan=plan,
        service_report=service_report,
        schedule_fixed=r1,
        schedule_rule=r2,
        schedule_opt=r3,
    )


def build_custom_scenario(
    name: str,
    solar_scale: float,
    demand_scale: float,
    storage_kwh: float,
    grid_available: bool,
    grid_capacity_kw: float,
) -> Scenario:
    return Scenario(
        name=name,
        description="User-defined custom scenario.",
        solar_scale=solar_scale,
        demand_scale=demand_scale,
        storage_kwh=storage_kwh,
        grid_available=grid_available,
        grid_capacity_kw=grid_capacity_kw,
        notes="Custom scenario — all values are illustrative.",
    )
