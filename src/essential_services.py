"""
essential_services.py — Essential Services Protection Layer (Module B).

Rules
-----
- Critical loads: hard constraints that must be met.
  If supply < critical minimum, report shortage explicitly.
- Important loads: schedulable within safe windows.
- Flexible loads: deferrable.

This module evaluates coverage and flags shortages.
It never silently downgrades critical loads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Facility definition
# ---------------------------------------------------------------------------

@dataclass
class Facility:
    key: str
    label: str
    priority: str          # "critical" | "important" | "flexible"
    min_energy_kwh_day: float
    operating_hours: tuple[int, int]  # (start_h, end_h) inclusive
    peak_demand_kw: float
    is_flexible: bool = False
    scheduling_window: Optional[tuple[int, int]] = None
    demand_col: Optional[str] = None   # column name in the dataset
    # Water-specific (optional)
    tank_capacity_litres: float = 0.0
    daily_demand_litres: float = 0.0
    pump_efficiency_kwh_per_kl: float = 0.5


DEFAULT_FACILITIES: list[Facility] = [
    Facility(
        key="health_centre",
        label="Primary Health Centre",
        priority="critical",
        min_energy_kwh_day=30.0,
        operating_hours=(0, 24),
        peak_demand_kw=5.0,
        is_flexible=False,
        demand_col="health_centre_demand_kw",
    ),
    Facility(
        key="water_facility",
        label="Drinking-Water Facility",
        priority="critical",
        min_energy_kwh_day=20.0,
        operating_hours=(6, 20),
        peak_demand_kw=6.0,
        is_flexible=True,
        scheduling_window=(6, 20),
        demand_col="water_facility_demand_kw",
        tank_capacity_litres=50_000.0,
        daily_demand_litres=30_000.0,
        pump_efficiency_kwh_per_kl=0.5,
    ),
    Facility(
        key="school",
        label="Community School",
        priority="important",
        min_energy_kwh_day=5.0,
        operating_hours=(7, 17),
        peak_demand_kw=4.0,
        is_flexible=True,
        scheduling_window=(6, 18),
        demand_col="school_demand_kw",
    ),
]


# ---------------------------------------------------------------------------
# Daily coverage analysis
# ---------------------------------------------------------------------------

@dataclass
class DailyCoverage:
    date: pd.Timestamp
    facility_key: str
    facility_label: str
    priority: str
    required_kwh: float
    available_kwh: float
    met_kwh: float
    shortage_kwh: float
    coverage_pct: float
    is_critical_shortage: bool
    warning_message: str = ""


@dataclass
class ServiceProtectionReport:
    daily_coverage: list[DailyCoverage] = field(default_factory=list)
    critical_shortages: list[DailyCoverage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_critical_shortage(self) -> bool:
        return len(self.critical_shortages) > 0

    def summary_df(self) -> pd.DataFrame:
        if not self.daily_coverage:
            return pd.DataFrame()
        rows = []
        for c in self.daily_coverage:
            rows.append({
                "Date": c.date.date(),
                "Facility": c.facility_label,
                "Priority": c.priority.capitalize(),
                "Required (kWh)": round(c.required_kwh, 2),
                "Available (kWh)": round(c.available_kwh, 2),
                "Met (kWh)": round(c.met_kwh, 2),
                "Shortage (kWh)": round(c.shortage_kwh, 2),
                "Coverage (%)": round(c.coverage_pct, 1),
                "Status": "⚠ SHORTAGE" if c.is_critical_shortage else "✓ Met",
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_service_coverage(
    forecast_df: pd.DataFrame,
    facilities: list[Facility],
    solar_scale: float = 1.0,
    demand_scale: float = 1.0,
) -> ServiceProtectionReport:
    """
    Evaluate daily essential-service coverage against forecast supply.

    Parameters
    ----------
    forecast_df:
        7-day hourly forecast with columns 'forecast_solar_kw',
        'forecast_demand_kw', and optionally per-facility demand columns.
    facilities:
        List of Facility objects.
    solar_scale, demand_scale:
        Scenario multipliers.

    Returns
    -------
    ServiceProtectionReport
    """
    report = ServiceProtectionReport()

    if forecast_df is None or len(forecast_df) == 0:
        report.warnings.append("No forecast data available for service-coverage evaluation.")
        return report

    # Daily totals from forecast
    daily_solar = (
        forecast_df["forecast_solar_kw"] * solar_scale
    ).resample("D").sum()  # kWh per day (hourly data → sum = kWh)

    daily_demand = (
        forecast_df["forecast_demand_kw"] * demand_scale
    ).resample("D").sum()

    # Available energy ≈ solar (grid/storage added in planner)
    # For pure coverage check: available = solar_kwh
    available_by_day = daily_solar

    for date in daily_solar.index:
        day_fc = forecast_df[forecast_df.index.normalize() == date]
        avail_kwh = float(available_by_day.get(date, 0.0))

        # Distribute available energy proportionally across critical first,
        # then important, then flexible.
        remaining = avail_kwh

        for priority in ("critical", "important", "flexible"):
            priority_facilities = [f for f in facilities if f.priority == priority]
            for fac in priority_facilities:
                req = fac.min_energy_kwh_day

                # Operational hours check
                op_h = fac.operating_hours[1] - fac.operating_hours[0]
                if op_h <= 0:
                    op_h = 24

                met = min(req, remaining)
                shortage = max(0.0, req - met)
                remaining = max(0.0, remaining - met)
                cov_pct = (met / req * 100) if req > 0 else 100.0

                is_critical_shortage = (
                    shortage > 0 and fac.priority == "critical"
                )

                msg = ""
                if is_critical_shortage:
                    msg = (
                        f"CRITICAL SHORTAGE on {date.date()}: "
                        f"{fac.label} requires {req:.1f} kWh but only "
                        f"{met:.1f} kWh available. Deficit: {shortage:.1f} kWh."
                    )
                    report.warnings.append(msg)

                cov = DailyCoverage(
                    date=date,
                    facility_key=fac.key,
                    facility_label=fac.label,
                    priority=fac.priority,
                    required_kwh=req,
                    available_kwh=avail_kwh,
                    met_kwh=met,
                    shortage_kwh=shortage,
                    coverage_pct=cov_pct,
                    is_critical_shortage=is_critical_shortage,
                    warning_message=msg,
                )
                report.daily_coverage.append(cov)
                if is_critical_shortage:
                    report.critical_shortages.append(cov)

    if not report.has_critical_shortage:
        report.warnings.append("")  # empty = no critical issues

    return report


# ---------------------------------------------------------------------------
# Water volume helper
# ---------------------------------------------------------------------------

def estimate_water_volume(
    energy_kwh: float,
    pump_efficiency_kwh_per_kl: float = 0.5,
) -> float:
    """
    Estimate water volume pumped from electrical energy.

    ASSUMPTIONS (illustrative, not validated engineering values):
    - pump_efficiency_kwh_per_kl: kWh needed to pump 1 000 litres.
    - Does NOT account for head pressure, pipe losses or groundwater depth.
    - Real water-supply systems require expert hydraulic assessment.

    Returns litres pumped.
    """
    if pump_efficiency_kwh_per_kl <= 0:
        raise ValueError("pump_efficiency_kwh_per_kl must be positive.")
    kl_pumped = energy_kwh / pump_efficiency_kwh_per_kl
    return kl_pumped * 1_000  # litres


def facility_from_dict(d: dict) -> Facility:
    """Build a Facility from a UI-edited dictionary."""
    return Facility(
        key=d.get("key", "custom"),
        label=d.get("label", "Custom Facility"),
        priority=d.get("priority", "flexible"),
        min_energy_kwh_day=float(d.get("min_energy_kwh_day", 5.0)),
        operating_hours=(
            int(d.get("op_start", 0)),
            int(d.get("op_end", 24)),
        ),
        peak_demand_kw=float(d.get("peak_demand_kw", 2.0)),
        is_flexible=bool(d.get("is_flexible", True)),
        scheduling_window=(
            int(d.get("sched_start", 6)),
            int(d.get("sched_end", 20)),
        ) if d.get("is_flexible") else None,
        demand_col=d.get("demand_col"),
        tank_capacity_litres=float(d.get("tank_capacity_litres", 0)),
        daily_demand_litres=float(d.get("daily_demand_litres", 0)),
        pump_efficiency_kwh_per_kl=float(d.get("pump_efficiency_kwh_per_kl", 0.5)),
    )
