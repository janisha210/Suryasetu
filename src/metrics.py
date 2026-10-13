"""
metrics.py — Energy balance, anomaly detection and summary statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Energy balance
# ---------------------------------------------------------------------------

def energy_balance_summary(
    forecast_df: pd.DataFrame,
    solar_scale: float = 1.0,
    demand_scale: float = 1.0,
) -> dict:
    """Return a dict of key weekly energy balance metrics."""
    solar = forecast_df["forecast_solar_kw"] * solar_scale
    demand = forecast_df["forecast_demand_kw"] * demand_scale
    net = solar - demand

    total_solar = float(solar.sum())
    total_demand = float(demand.sum())
    surplus = float(net.clip(lower=0).sum())
    deficit = float((-net).clip(lower=0).sum())
    self_sufficiency = (
        min(total_solar, total_demand) / total_demand * 100
        if total_demand > 0 else 0.0
    )

    return {
        "total_solar_kwh": round(total_solar, 2),
        "total_demand_kwh": round(total_demand, 2),
        "net_surplus_kwh": round(surplus, 2),
        "net_deficit_kwh": round(deficit, 2),
        "self_sufficiency_pct": round(self_sufficiency, 1),
        "peak_solar_kw": round(float(solar.max()), 2),
        "peak_demand_kw": round(float(demand.max()), 2),
        "avg_solar_kw": round(float(solar.mean()), 3),
        "avg_demand_kw": round(float(demand.mean()), 3),
    }


# ---------------------------------------------------------------------------
# Anomaly detection (simple threshold-based, Module F)
# ---------------------------------------------------------------------------

def detect_anomalies(
    df: pd.DataFrame,
    column: str,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Flag values more than *z_threshold* standard deviations from the
    rolling 24-hour mean as potential anomalies.

    NOTE: This is a simple statistical check on SYNTHETIC data.
    It does NOT detect real equipment faults. For real deployment,
    validated sensor data and domain-expert thresholds are required.

    Returns a DataFrame with an 'anomaly' boolean column.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    series = df[column].copy()
    rolling_mean = series.rolling(window=24, min_periods=1).mean()
    rolling_std = series.rolling(window=24, min_periods=1).std().fillna(0)

    z_scores = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    anomalies = (z_scores.abs() > z_threshold).fillna(False)

    result = df[[column]].copy()
    result["rolling_mean"] = rolling_mean
    result["z_score"] = z_scores
    result["anomaly"] = anomalies
    result["data_source"] = "Synthetic ⚠"
    return result


# ---------------------------------------------------------------------------
# Coverage statistics
# ---------------------------------------------------------------------------

def compute_coverage_stats(
    daily_coverage_df: pd.DataFrame,
) -> dict:
    """Aggregate coverage statistics from the service protection report."""
    if daily_coverage_df is None or len(daily_coverage_df) == 0:
        return {}

    critical = daily_coverage_df[daily_coverage_df["Priority"] == "Critical"]
    important = daily_coverage_df[daily_coverage_df["Priority"] == "Important"]

    stats = {
        "critical_avg_coverage_pct": (
            float(critical["Coverage (%)"].mean()) if len(critical) else 100.0
        ),
        "important_avg_coverage_pct": (
            float(important["Coverage (%)"].mean()) if len(important) else 100.0
        ),
        "n_critical_shortage_days": int(
            (daily_coverage_df["Shortage (kWh)"] > 0.1).sum()
        ),
        "total_shortage_kwh": float(daily_coverage_df["Shortage (kWh)"].sum()),
    }
    return stats
