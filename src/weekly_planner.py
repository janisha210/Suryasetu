"""
weekly_planner.py — Seven-day energy planner (Module C).

Builds a rolling 7-day plan from forecast data and facility schedules.
Shows daily energy balance, critical reserves, surpluses/shortages and
recommended flexible-load timings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from essential_services import Facility


# ---------------------------------------------------------------------------
# Hourly plan row
# ---------------------------------------------------------------------------

@dataclass
class WeeklyPlan:
    hourly_df: pd.DataFrame      # hour-level detail
    daily_df: pd.DataFrame       # day-level summary
    flex_recommendations: list[dict]
    shortage_warnings: list[str]
    critical_reserve_kwh_day: float
    total_surplus_kwh: float
    total_shortage_kwh: float
    grid_energy_kwh: float
    renewable_fraction: float     # 0–1

    def daily_summary_display(self) -> pd.DataFrame:
        d = self.daily_df.copy()
        d.index = d.index.date
        d.index.name = "Date"
        d.columns = [c.replace("_", " ").title() for c in d.columns]
        return d.round(2)


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------

def build_weekly_plan(
    forecast_df: pd.DataFrame,
    facilities: list[Facility],
    solar_scale: float = 1.0,
    demand_scale: float = 1.0,
    grid_capacity_kw: float = 5.0,
    storage_kwh: float = 0.0,
    grid_available: bool = True,
) -> WeeklyPlan:
    """
    Build a 7-day energy plan.

    Parameters
    ----------
    forecast_df:
        7-day hourly forecast DataFrame from ForecastPipeline.
    facilities:
        List of Facility objects.
    solar_scale, demand_scale:
        Scenario multipliers.
    grid_capacity_kw:
        Maximum hourly grid import (kW). Set to 0 for outage scenario.
    storage_kwh:
        Battery storage capacity (kWh). 0 = no storage.
    grid_available:
        If False, grid_capacity_kw is forced to 0.

    Returns
    -------
    WeeklyPlan
    """
    if forecast_df is None or len(forecast_df) == 0:
        raise ValueError("forecast_df is empty; cannot build plan.")

    if not grid_available:
        grid_capacity_kw = 0.0

    solar = (forecast_df["forecast_solar_kw"] * solar_scale).values
    demand = (forecast_df["forecast_demand_kw"] * demand_scale).values
    T = len(solar)

    hours = forecast_df.index.hour.values
    dates = forecast_df.index.normalize()

    # Storage state
    soc = storage_kwh * 0.5       # start at half charge
    max_soc = max(storage_kwh, 0.0)
    eta = 0.9                      # round-trip efficiency

    solar_used = np.zeros(T)
    grid_used = np.zeros(T)
    stor_charge = np.zeros(T)
    stor_discharge = np.zeros(T)
    stor_soc_arr = np.zeros(T)
    unmet = np.zeros(T)
    surplus_arr = np.zeros(T)

    for t in range(T):
        avail = solar[t]
        needed = demand[t]

        used = min(avail, needed)
        solar_used[t] = used
        surplus = avail - used
        deficit = needed - used

        # Charge storage with surplus
        charge = min(surplus * eta, max_soc - soc)
        soc += charge
        stor_charge[t] = charge
        surplus -= charge / eta

        # Discharge for deficit
        discharge = min(deficit / eta, soc)
        soc -= discharge
        stor_discharge[t] = discharge * eta
        deficit -= stor_discharge[t]

        # Grid supply
        if grid_capacity_kw > 0:
            grid = min(deficit, grid_capacity_kw)
            grid_used[t] = grid
            deficit -= grid

        unmet[t] = max(0.0, deficit)
        surplus_arr[t] = max(0.0, avail - needed - charge / eta)
        stor_soc_arr[t] = soc

    # Build hourly DataFrame
    hourly = pd.DataFrame(
        {
            "solar_kw": solar,
            "demand_kw": demand,
            "solar_used_kw": solar_used,
            "grid_kw": grid_used,
            "storage_charge_kw": stor_charge,
            "storage_discharge_kw": stor_discharge,
            "storage_soc_kwh": stor_soc_arr,
            "surplus_kw": surplus_arr,
            "unmet_kw": unmet,
            "net_balance_kw": solar - demand,
        },
        index=forecast_df.index,
    )

    # Daily summary
    daily = pd.DataFrame(
        {
            "solar_kwh": pd.Series(solar, index=forecast_df.index).resample("D").sum(),
            "demand_kwh": pd.Series(demand, index=forecast_df.index).resample("D").sum(),
            "grid_kwh": pd.Series(grid_used, index=forecast_df.index).resample("D").sum(),
            "unmet_kwh": pd.Series(unmet, index=forecast_df.index).resample("D").sum(),
            "surplus_kwh": pd.Series(surplus_arr, index=forecast_df.index).resample("D").sum(),
            "solar_used_kwh": pd.Series(solar_used, index=forecast_df.index).resample("D").sum(),
        }
    )

    # Critical reserve requirement per day
    crit_req_per_day = sum(
        f.min_energy_kwh_day for f in facilities if f.priority == "critical"
    )

    # Shortage warnings
    warnings = []
    total_shortage = float(unmet.sum())
    if total_shortage > 0.1:
        warnings.append(
            f"⚠ Total unmet demand over 7 days: {total_shortage:.1f} kWh. "
            "Critical service coverage may be insufficient."
        )

    for date, row in daily.iterrows():
        if row["unmet_kwh"] > 0.1:
            day_crit_avail = max(0, row["solar_kwh"] + row["grid_kwh"] - row["unmet_kwh"])
            if day_crit_avail < crit_req_per_day:
                warnings.append(
                    f"⚠ {date.date()}: Available energy {row['solar_kwh']:.1f} kWh "
                    f"may not cover critical requirement of {crit_req_per_day:.1f} kWh."
                )

    if not grid_available:
        warnings.append(
            "ℹ Grid outage scenario: All energy must come from solar and storage."
        )

    # Flexible-load recommendations
    flex_recs = _generate_flex_recommendations(forecast_df, facilities, solar, demand, solar_scale, demand_scale)

    total_solar = float(solar_used.sum())
    total_energy = total_solar + float(grid_used.sum())
    ren_frac = total_solar / total_energy if total_energy > 0 else 0.0

    return WeeklyPlan(
        hourly_df=hourly,
        daily_df=daily,
        flex_recommendations=flex_recs,
        shortage_warnings=warnings,
        critical_reserve_kwh_day=crit_req_per_day,
        total_surplus_kwh=float(surplus_arr.sum()),
        total_shortage_kwh=total_shortage,
        grid_energy_kwh=float(grid_used.sum()),
        renewable_fraction=ren_frac,
    )


def _generate_flex_recommendations(
    forecast_df: pd.DataFrame,
    facilities: list[Facility],
    solar: np.ndarray,
    demand: np.ndarray,
    solar_scale: float,
    demand_scale: float,
) -> list[dict]:
    """Generate human-readable flexible-load scheduling recommendations."""
    recs = []
    net = solar - demand
    hours = forecast_df.index.hour.values
    dates = forecast_df.index.normalize()

    for fac in facilities:
        if not fac.is_flexible:
            continue

        sw = fac.scheduling_window or fac.operating_hours
        sw_start, sw_end = sw

        for date in pd.unique(dates):
            day_mask = dates == date
            day_hours = hours[day_mask]
            day_net = net[day_mask]

            # Find best hours in window with surplus solar
            window_mask = (day_hours >= sw_start) & (day_hours < sw_end)
            window_net = day_net * window_mask

            if window_net.max() > 0:
                best_idx = np.argmax(window_net)
                best_hour = day_hours[best_idx]
                surplus_kw = window_net[best_idx]
                reason = (
                    f"Solar surplus of {surplus_kw:.1f} kW expected around "
                    f"{best_hour:02d}:00–{(best_hour+1):02d}:00."
                )
            else:
                best_hour = sw_start
                reason = (
                    "No clear surplus window. Schedule at the start of the "
                    "operating window to minimise grid draw."
                )

            recs.append({
                "date": date.date(),
                "facility": fac.label,
                "priority": fac.priority,
                "recommended_start_h": int(best_hour),
                "scheduling_window": f"{sw_start:02d}:00–{sw_end:02d}:00",
                "reason": reason,
            })

    return recs
