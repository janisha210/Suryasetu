"""
optimiser.py — Energy Optimisation Engine (Module D).

Uses PuLP to optimise the 7-day hourly schedule.

Three scheduling approaches are compared on identical scenario assumptions:
  1. Fixed schedule   — loads run at their default operating hours regardless.
  2. Rule-based       — shift flexible loads to hours with surplus solar.
  3. Optimised (PuLP) — minimise unmet demand + grid consumption while
                        respecting hard constraints for critical loads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

from essential_services import Facility


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScheduleResult:
    method: str
    hourly_df: pd.DataFrame            # hour-level schedule
    daily_df: pd.DataFrame             # day-level summary
    total_unmet_kwh: float
    total_grid_kwh: float
    total_solar_used_kwh: float
    critical_coverage_pct: float
    feasible: bool
    infeasibility_notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict:
        return {
            "Method": self.method,
            "Total Unmet Demand (kWh)": round(self.total_unmet_kwh, 2),
            "Grid Energy Used (kWh)": round(self.total_grid_kwh, 2),
            "Solar Utilised (kWh)": round(self.total_solar_used_kwh, 2),
            "Critical Coverage (%)": round(self.critical_coverage_pct, 1),
            "Feasible": "Yes" if self.feasible else "No ⚠",
        }


# ---------------------------------------------------------------------------
# Helper: build hourly demand from forecast + facility config
# ---------------------------------------------------------------------------

def _build_hourly_demand(
    forecast_df: pd.DataFrame,
    facilities: list[Facility],
    demand_scale: float,
) -> pd.DataFrame:
    """Return per-facility hourly demand (kW) for the forecast window."""
    n = len(forecast_df)
    hours = forecast_df.index.hour.values
    dow = forecast_df.index.dayofweek.values

    result = pd.DataFrame(index=forecast_df.index)

    for fac in facilities:
        col = fac.demand_col
        if col and col in forecast_df.columns:
            result[fac.key] = forecast_df[col].values * demand_scale
        else:
            # Reconstruct from total proportionally
            demand_total = forecast_df["forecast_demand_kw"].values * demand_scale
            n_facilities = len(facilities)
            result[fac.key] = demand_total / n_facilities

        # Zero out demand outside operating hours
        op_start, op_end = fac.operating_hours
        mask = ~((hours >= op_start) & (hours < op_end))
        if op_start == 0 and op_end == 24:
            mask[:] = False  # 24-h facility
        result.loc[mask, fac.key] = 0.0

    return result


# ---------------------------------------------------------------------------
# Method 1: Fixed schedule
# ---------------------------------------------------------------------------

def fixed_schedule(
    forecast_df: pd.DataFrame,
    facilities: list[Facility],
    solar_scale: float = 1.0,
    demand_scale: float = 1.0,
    grid_capacity_kw: float = 5.0,
    storage_kwh: float = 0.0,
) -> ScheduleResult:
    """Loads run at their configured operating hours; no shifting."""
    solar = forecast_df["forecast_solar_kw"].values * solar_scale
    demand_df = _build_hourly_demand(forecast_df, facilities, demand_scale)
    total_demand = demand_df.sum(axis=1).values

    n = len(solar)
    storage = storage_kwh * 0.5  # start at half charge
    max_storage = storage_kwh

    grid_used = np.zeros(n)
    solar_used = np.zeros(n)
    unmet = np.zeros(n)
    storage_soc = np.zeros(n)

    for i in range(n):
        avail = solar[i]
        needed = total_demand[i]

        # Use solar first
        used_solar = min(avail, needed)
        solar_used[i] = used_solar
        surplus = avail - used_solar
        deficit = needed - used_solar

        # Charge storage with surplus
        charge = min(surplus, max_storage - storage)
        storage += charge

        # Discharge storage for deficit
        discharge = min(deficit, storage)
        storage -= discharge
        deficit -= discharge

        # Grid supply
        grid = min(deficit, grid_capacity_kw)
        grid_used[i] = grid
        deficit -= grid

        unmet[i] = max(0.0, deficit)
        storage_soc[i] = storage

    hourly = forecast_df[["forecast_solar_kw", "forecast_demand_kw"]].copy()
    hourly["demand_kw"] = total_demand
    hourly["solar_used_kw"] = solar_used
    hourly["grid_kw"] = grid_used
    hourly["unmet_kw"] = unmet
    hourly["storage_soc_kwh"] = storage_soc
    hourly["method"] = "Fixed"

    daily = _daily_summary(hourly, forecast_df, solar_scale)

    crit_req = sum(
        f.min_energy_kwh_day * 7 for f in facilities if f.priority == "critical"
    )
    crit_met = max(0.0, crit_req - float(unmet.sum()))
    cov = (crit_met / crit_req * 100) if crit_req > 0 else 100.0

    return ScheduleResult(
        method="1. Fixed Schedule",
        hourly_df=hourly,
        daily_df=daily,
        total_unmet_kwh=float(unmet.sum()),
        total_grid_kwh=float(grid_used.sum()),
        total_solar_used_kwh=float(solar_used.sum()),
        critical_coverage_pct=min(100.0, cov),
        feasible=float(unmet.sum()) < 1.0,
    )


# ---------------------------------------------------------------------------
# Method 2: Rule-based schedule
# ---------------------------------------------------------------------------

def rule_based_schedule(
    forecast_df: pd.DataFrame,
    facilities: list[Facility],
    solar_scale: float = 1.0,
    demand_scale: float = 1.0,
    grid_capacity_kw: float = 5.0,
    storage_kwh: float = 0.0,
) -> ScheduleResult:
    """
    Shift flexible loads to surplus-solar hours.
    Rule: If forecasted solar > non-flexible demand in an hour,
    schedule flexible load in that hour instead of its default window.
    """
    solar = forecast_df["forecast_solar_kw"].values * solar_scale
    demand_df = _build_hourly_demand(forecast_df, facilities, demand_scale)

    # Separate critical (non-shiftable) and flexible demand
    critical_demand = np.zeros(len(solar))
    flexible_pools: dict[str, tuple[np.ndarray, Facility]] = {}

    for fac in facilities:
        col_vals = demand_df[fac.key].values
        if not fac.is_flexible or fac.priority == "critical":
            critical_demand += col_vals
        else:
            flexible_pools[fac.key] = (col_vals, fac)

    # For each flexible facility: compute daily total; spread over surplus hours
    scheduled_flex = np.zeros(len(solar))
    hours = forecast_df.index.hour.values
    dates = forecast_df.index.normalize()

    for fac_key, (orig_demand, fac) in flexible_pools.items():
        daily_groups = pd.Series(orig_demand, index=forecast_df.index).resample("D")
        for date, day_series in daily_groups:
            daily_total = day_series.sum()
            if daily_total <= 0:
                continue

            # Hours within scheduling window with solar surplus
            day_mask = dates == date
            day_solar = solar[day_mask]
            day_crit = critical_demand[day_mask]
            day_hours = hours[day_mask]

            if fac.scheduling_window:
                sw_start, sw_end = fac.scheduling_window
                window_mask = (day_hours >= sw_start) & (day_hours < sw_end)
            else:
                op_s, op_e = fac.operating_hours
                window_mask = (day_hours >= op_s) & (day_hours < op_e)

            surplus = np.maximum(0, day_solar - day_crit)
            surplus_in_window = surplus * window_mask

            total_surplus = surplus_in_window.sum()
            if total_surplus > 0:
                # Distribute flex demand proportionally to surplus
                distributed = (surplus_in_window / total_surplus) * daily_total
            else:
                # Fall back to original schedule
                orig_day = orig_demand[day_mask]
                distributed = orig_day

            # Place in scheduled_flex
            day_indices = np.where(day_mask)[0]
            scheduled_flex[day_indices] += distributed

    total_demand = critical_demand + scheduled_flex

    n = len(solar)
    storage = storage_kwh * 0.5
    max_storage = storage_kwh
    grid_used = np.zeros(n)
    solar_used = np.zeros(n)
    unmet = np.zeros(n)
    storage_soc = np.zeros(n)

    for i in range(n):
        avail = solar[i]
        needed = total_demand[i]
        used_solar = min(avail, needed)
        solar_used[i] = used_solar
        surplus = avail - used_solar
        deficit = needed - used_solar
        charge = min(surplus, max_storage - storage)
        storage += charge
        discharge = min(deficit, storage)
        storage -= discharge
        deficit -= discharge
        grid = min(deficit, grid_capacity_kw)
        grid_used[i] = grid
        deficit -= grid
        unmet[i] = max(0.0, deficit)
        storage_soc[i] = storage

    hourly = forecast_df[["forecast_solar_kw", "forecast_demand_kw"]].copy()
    hourly["demand_kw"] = total_demand
    hourly["solar_used_kw"] = solar_used
    hourly["grid_kw"] = grid_used
    hourly["unmet_kw"] = unmet
    hourly["storage_soc_kwh"] = storage_soc
    hourly["method"] = "Rule-based"

    daily = _daily_summary(hourly, forecast_df, solar_scale)

    crit_req = sum(
        f.min_energy_kwh_day * 7 for f in facilities if f.priority == "critical"
    )
    crit_met = max(0.0, crit_req - float(unmet.sum()))
    cov = (crit_met / crit_req * 100) if crit_req > 0 else 100.0

    return ScheduleResult(
        method="2. Rule-Based Schedule",
        hourly_df=hourly,
        daily_df=daily,
        total_unmet_kwh=float(unmet.sum()),
        total_grid_kwh=float(grid_used.sum()),
        total_solar_used_kwh=float(solar_used.sum()),
        critical_coverage_pct=min(100.0, cov),
        feasible=float(unmet.sum()) < 1.0,
    )


# ---------------------------------------------------------------------------
# Method 3: PuLP optimised schedule
# ---------------------------------------------------------------------------

def optimised_schedule(
    forecast_df: pd.DataFrame,
    facilities: list[Facility],
    solar_scale: float = 1.0,
    demand_scale: float = 1.0,
    grid_capacity_kw: float = 5.0,
    storage_kwh: float = 0.0,
) -> ScheduleResult:
    """
    Minimise: unmet_demand + 0.5 * grid_usage
    Subject to:
    - Critical facilities must be served (hard constraint).
    - Flexible loads may be shifted within their windows.
    - Storage charge/discharge balance.
    - Grid capacity limit.
    """
    if not PULP_AVAILABLE:
        notes = ["PuLP not available; falling back to rule-based schedule."]
        result = rule_based_schedule(
            forecast_df, facilities, solar_scale, demand_scale,
            grid_capacity_kw, storage_kwh,
        )
        result.method = "3. Optimised (PuLP fallback → Rule-based)"
        result.infeasibility_notes = notes
        return result

    solar = forecast_df["forecast_solar_kw"].values * solar_scale
    demand_df = _build_hourly_demand(forecast_df, facilities, demand_scale)
    T = len(solar)

    prob = pulp.LpProblem("SuryaSetu_Weekly_Schedule", pulp.LpMinimize)

    # Decision variables
    grid = [prob.add_variable(f"grid_{t}", lowBound=0, upBound=grid_capacity_kw) for t in range(T)]
    unmet = [prob.add_variable(f"unmet_{t}", lowBound=0) for t in range(T)]
    stor_charge = [prob.add_variable(f"chg_{t}", lowBound=0) for t in range(T)]
    stor_discharge = [prob.add_variable(f"dis_{t}", lowBound=0) for t in range(T)]
    stor_soc = [prob.add_variable(f"soc_{t}", lowBound=0, upBound=max(storage_kwh, 0.001)) for t in range(T)]

    # Flexible load variables (per facility per hour)
    flex_vars: dict[str, list] = {}
    for fac in facilities:
        if fac.is_flexible and fac.priority != "critical":
            # Daily total must match original daily demand
            daily_demand = demand_df[fac.key].resample("D").sum()
            flex_vars[fac.key] = []
            hours_arr = forecast_df.index.hour.values
            dates_arr = forecast_df.index.normalize()

            for t in range(T):
                h = hours_arr[t]
                if fac.scheduling_window:
                    sw_s, sw_e = fac.scheduling_window
                    in_window = sw_s <= h < sw_e
                else:
                    op_s, op_e = fac.operating_hours
                    in_window = op_s <= h < op_e
                # Max bound: peak demand
                ub = fac.peak_demand_kw if in_window else 0
                v = prob.add_variable(
                    f"flex_{fac.key}_{t}", lowBound=0, upBound=ub
                )
                flex_vars[fac.key].append(v)

            # Daily total constraint
            for date, d_kwh in daily_demand.items():
                day_mask = [i for i, d in enumerate(dates_arr) if d == date]
                if not day_mask:
                    continue
                day_flex = [flex_vars[fac.key][i] for i in day_mask]
                # Allow ±5% tolerance
                prob += pulp.lpSum(day_flex) >= d_kwh * 0.95
                prob += pulp.lpSum(day_flex) <= d_kwh * 1.05 + 0.001

    # Objective: minimise unmet + 0.5 * grid
    prob += pulp.lpSum(unmet) + 0.5 * pulp.lpSum(grid)

    max_stor = max(storage_kwh, 0.001)
    init_soc = storage_kwh * 0.5

    for t in range(T):
        # Non-critical fixed demand
        fixed_d = sum(
            demand_df[fac.key].iloc[t]
            for fac in facilities
            if not fac.is_flexible or fac.priority == "critical"
        )
        # Flexible demand at this hour
        flex_d = pulp.lpSum(
            flex_vars[fac.key][t]
            for fac in facilities
            if fac.is_flexible and fac.priority != "critical" and fac.key in flex_vars
        )
        total_d = fixed_d + flex_d

        # Energy balance: solar + grid + discharge = demand + charge + unmet_adjustment
        prob += (
            solar[t] + grid[t] + stor_discharge[t]
            == total_d - unmet[t] + stor_charge[t]
        )

        # Charge/discharge limits
        prob += stor_charge[t] <= max_stor
        prob += stor_discharge[t] <= max_stor

        # SoC continuity
        if t == 0:
            prob += stor_soc[t] == init_soc + stor_charge[t] - stor_discharge[t]
        else:
            prob += stor_soc[t] == stor_soc[t - 1] + stor_charge[t] - stor_discharge[t]

        prob += stor_soc[t] >= 0
        prob += stor_soc[t] <= max_stor

        # Critical facilities: unmet must be 0
        crit_demand = sum(
            demand_df[fac.key].iloc[t]
            for fac in facilities
            if fac.priority == "critical" and not fac.is_flexible
        )
        if crit_demand > 0:
            prob += unmet[t] <= total_d - crit_demand + 0.001

    # Select solver
    solver = None
    if hasattr(pulp, "HiGHS"):
        try:
            highs = pulp.HiGHS(msg=0)
            if highs.available():
                solver = highs
        except Exception:
            pass
    if solver is None and hasattr(pulp, "PULP_CBC_CMD"):
        try:
            solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=60)
        except Exception:
            pass

    solve_res = prob.solve(solver) if solver is not None else prob.solve()

    status_name = ""
    feasible = False
    if hasattr(solve_res, "status"):
        status_name = getattr(solve_res.status, "name", str(solve_res.status))
        feasible = status_name in ("Optimal", "Feasible") or solve_res.status == getattr(pulp.LpSolveStatus, "Optimal", 1)
    elif hasattr(prob, "status"):
        status_name = pulp.LpStatus.get(prob.status, str(prob.status)) if hasattr(pulp, "LpStatus") else str(prob.status)
        feasible = status_name in ("Optimal", "Feasible")

    notes = []
    if not feasible:
        notes.append(
            f"Optimiser status: {status_name}. "
            "The scenario may be infeasible — critical demand exceeds supply."
        )
        # Fall back to rule-based
        fallback = rule_based_schedule(
            forecast_df, facilities, solar_scale, demand_scale,
            grid_capacity_kw, storage_kwh,
        )
        fallback.method = "3. Optimised (infeasible → Rule-based fallback)"
        fallback.infeasibility_notes = notes
        fallback.feasible = False
        return fallback

    # Extract solution
    grid_vals = np.array([pulp.value(g) or 0.0 for g in grid])
    unmet_vals = np.array([pulp.value(u) or 0.0 for u in unmet])
    soc_vals = np.array([pulp.value(s) or 0.0 for s in stor_soc])

    solar_used = np.minimum(solar, solar - np.maximum(0, solar - (solar + grid_vals - unmet_vals)))
    solar_used = np.clip(solar_used, 0, solar)

    hourly = forecast_df[["forecast_solar_kw", "forecast_demand_kw"]].copy()
    hourly["demand_kw"] = demand_df.sum(axis=1).values
    hourly["solar_used_kw"] = solar_used
    hourly["grid_kw"] = np.clip(grid_vals, 0, None)
    hourly["unmet_kw"] = np.clip(unmet_vals, 0, None)
    hourly["storage_soc_kwh"] = np.clip(soc_vals, 0, None)
    hourly["method"] = "Optimised"

    daily = _daily_summary(hourly, forecast_df, solar_scale)

    crit_req = sum(
        f.min_energy_kwh_day * 7 for f in facilities if f.priority == "critical"
    )
    total_unmet = float(np.clip(unmet_vals, 0, None).sum())
    crit_met = max(0.0, crit_req - total_unmet)
    cov = (crit_met / crit_req * 100) if crit_req > 0 else 100.0

    return ScheduleResult(
        method="3. Forecast-Informed Optimised (PuLP)",
        hourly_df=hourly,
        daily_df=daily,
        total_unmet_kwh=total_unmet,
        total_grid_kwh=float(np.clip(grid_vals, 0, None).sum()),
        total_solar_used_kwh=float(solar_used.sum()),
        critical_coverage_pct=min(100.0, cov),
        feasible=True,
        infeasibility_notes=notes,
    )


# ---------------------------------------------------------------------------
# Daily summary helper
# ---------------------------------------------------------------------------

def _daily_summary(
    hourly: pd.DataFrame,
    forecast_df: pd.DataFrame,
    solar_scale: float,
) -> pd.DataFrame:
    daily = pd.DataFrame()
    daily["solar_kwh"] = (forecast_df["forecast_solar_kw"] * solar_scale).resample("D").sum()
    daily["demand_kwh"] = forecast_df["forecast_demand_kw"].resample("D").sum()
    daily["grid_kwh"] = hourly["grid_kw"].resample("D").sum()
    daily["unmet_kwh"] = hourly["unmet_kw"].resample("D").sum()
    daily["solar_used_kwh"] = hourly["solar_used_kw"].resample("D").sum()
    daily["surplus_kwh"] = (forecast_df["forecast_solar_kw"] * solar_scale - forecast_df["forecast_demand_kw"]).clip(lower=0).resample("D").sum()
    return daily


# ---------------------------------------------------------------------------
# Run all three and return comparison
# ---------------------------------------------------------------------------

def compare_schedules(
    forecast_df: pd.DataFrame,
    facilities: list[Facility],
    solar_scale: float = 1.0,
    demand_scale: float = 1.0,
    grid_capacity_kw: float = 5.0,
    storage_kwh: float = 0.0,
) -> tuple[ScheduleResult, ScheduleResult, ScheduleResult]:
    r1 = fixed_schedule(forecast_df, facilities, solar_scale, demand_scale, grid_capacity_kw, storage_kwh)
    r2 = rule_based_schedule(forecast_df, facilities, solar_scale, demand_scale, grid_capacity_kw, storage_kwh)
    r3 = optimised_schedule(forecast_df, facilities, solar_scale, demand_scale, grid_capacity_kw, storage_kwh)
    return r1, r2, r3
