"""
SuryaSetu — Synthetic Data Generator
======================================
Generates plausible but explicitly SYNTHETIC hourly energy data for three
illustrative facilities over 90 days.

All values are illustrative. They are not measured data from any real site.
Units: kWh (electrical energy), kW (power), litres (water volume).
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START_DATE = pd.Timestamp("2026-01-01")
N_DAYS = 90
HOURS = N_DAYS * 24

FACILITIES = {
    "health_centre": {
        "label": "Primary Health Centre",
        "priority": "critical",
        "base_demand_kw": 2.5,      # average baseload kW
        "peak_demand_kw": 5.0,      # weekday peak kW
        "operating_hours": (0, 24), # 24 h / day
        "min_energy_kwh_day": 30.0, # hard minimum per day
    },
    "water_facility": {
        "label": "Drinking-Water Facility",
        "priority": "critical",
        "base_demand_kw": 3.0,
        "peak_demand_kw": 6.0,
        "operating_hours": (6, 20), # pumps run 06:00–20:00
        "min_energy_kwh_day": 20.0,
        # Water-specific parameters (illustrative)
        "tank_capacity_litres": 50_000,
        "daily_demand_litres": 30_000,
        "pump_efficiency_kwh_per_kl": 0.5,  # kWh per 1 000 litres pumped
    },
    "school": {
        "label": "Community School",
        "priority": "important",
        "base_demand_kw": 1.0,
        "peak_demand_kw": 4.0,
        "operating_hours": (7, 17),  # school hours Mon–Fri; weekend lower
        "min_energy_kwh_day": 5.0,
    },
}

SOLAR_CAPACITY_KW = 15.0  # total PV capacity for the community


# ---------------------------------------------------------------------------
# Solar irradiance profile (synthetic)
# ---------------------------------------------------------------------------

def solar_profile(hour: int, doy: int) -> float:
    """Return a 0–1 irradiance factor.

    Uses a simple sinusoidal model peaking at solar noon (hour=12).
    Seasonal variation: slightly higher in summer months (DOY ~172).
    """
    # Sunrise at ~6 h, sunset at ~18 h on equinox; seasonal ±1 h shift
    season_shift = np.sin(2 * np.pi * (doy - 172) / 365)  # -1..+1
    sunrise = 6.0 - season_shift
    sunset = 18.0 + season_shift

    if hour < sunrise or hour >= sunset:
        return 0.0

    solar_noon = (sunrise + sunset) / 2
    half_day = (sunset - sunrise) / 2
    factor = np.sin(np.pi * (hour - sunrise) / (sunset - sunrise))
    # Peak is slightly above 1 at noon; clip to 1
    return float(np.clip(factor, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Demand profile helpers
# ---------------------------------------------------------------------------

def health_centre_demand(hour: int, dow: int) -> float:
    """kW demand for primary health centre (24 h facility)."""
    # Night baseload, daytime clinic peak
    if 0 <= hour < 6:
        return 1.5 + rng.normal(0, 0.1)
    if 6 <= hour < 9:
        return 2.0 + rng.normal(0, 0.2)
    if 9 <= hour < 17:
        return 5.0 + rng.normal(0, 0.4)   # clinic hours
    if 17 <= hour < 21:
        return 3.5 + rng.normal(0, 0.3)
    return 2.0 + rng.normal(0, 0.15)


def water_facility_demand(hour: int, dow: int) -> float:
    """kW demand for water-pumping facility (operating 06:00–20:00)."""
    if hour < 6 or hour >= 20:
        return 0.2  # minimal standby
    if 6 <= hour < 9:
        return 5.5 + rng.normal(0, 0.3)  # morning fill
    if 9 <= hour < 17:
        return 3.0 + rng.normal(0, 0.4)
    if 17 <= hour < 20:
        return 4.0 + rng.normal(0, 0.3)  # evening fill
    return 0.2


def school_demand(hour: int, dow: int) -> float:
    """kW demand for school / community building."""
    is_weekend = dow >= 5
    if is_weekend:
        # Weekend community use: light load
        if 8 <= hour < 18:
            return 0.8 + rng.normal(0, 0.15)
        return 0.1
    # Weekday school
    if hour < 7 or hour >= 17:
        return 0.1
    if 7 <= hour < 9:
        return 1.5 + rng.normal(0, 0.2)
    if 9 <= hour < 12:
        return 3.5 + rng.normal(0, 0.3)
    if 12 <= hour < 13:
        return 4.0 + rng.normal(0, 0.3)  # midday peak (cooking)
    if 13 <= hour < 15:
        return 3.0 + rng.normal(0, 0.3)
    if 15 <= hour < 17:
        return 2.5 + rng.normal(0, 0.25)
    return 0.1


DEMAND_FN = {
    "health_centre": health_centre_demand,
    "water_facility": water_facility_demand,
    "school": school_demand,
}


# ---------------------------------------------------------------------------
# Cloud & weather variability
# ---------------------------------------------------------------------------

def cloud_factor(doy: int) -> float:
    """Daily cloud-cover factor: 0 (full cloud) to 1 (clear sky)."""
    # Monsoon-like dip around July–September (DOY 180–270)
    monsoon = np.sin(np.pi * max(0, min(doy - 180, 90)) / 90)
    base = 0.85 - 0.35 * monsoon
    return float(np.clip(base + rng.normal(0, 0.08), 0.1, 1.0))


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_dataset() -> pd.DataFrame:
    rows = []
    timestamps = pd.date_range(START_DATE, periods=HOURS, freq="h")

    for ts in timestamps:
        h = ts.hour
        dow = ts.dayofweek   # 0=Mon .. 6=Sun
        doy = ts.dayofyear

        cf = cloud_factor(doy)
        irr = solar_profile(h, doy)
        solar_kw = SOLAR_CAPACITY_KW * irr * cf * (1 + rng.normal(0, 0.03))
        solar_kw = float(np.clip(solar_kw, 0, SOLAR_CAPACITY_KW))

        hc = float(np.clip(health_centre_demand(h, dow), 0, 10))
        wf = float(np.clip(water_facility_demand(h, dow), 0, 10))
        sc = float(np.clip(school_demand(h, dow), 0, 8))
        total_demand = hc + wf + sc

        rows.append({
            "timestamp": ts,
            "hour": h,
            "day_of_week": dow,
            "day_of_year": doy,
            "cloud_factor": round(cf, 4),
            "solar_irradiance_factor": round(irr, 4),
            "solar_generation_kw": round(solar_kw, 3),
            "health_centre_demand_kw": round(hc, 3),
            "water_facility_demand_kw": round(wf, 3),
            "school_demand_kw": round(sc, 3),
            "total_demand_kw": round(total_demand, 3),
            "net_balance_kw": round(solar_kw - total_demand, 3),
        })

    df = pd.DataFrame(rows)
    # Add lagged features (useful for ML)
    df["solar_lag1h"] = df["solar_generation_kw"].shift(1)
    df["demand_lag1h"] = df["total_demand_kw"].shift(1)
    df["solar_lag24h"] = df["solar_generation_kw"].shift(24)
    df["demand_lag24h"] = df["total_demand_kw"].shift(24)
    return df


def save_dataset(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    path = output_dir / "synthetic_energy_data.csv"
    df.to_csv(path, index=False)
    print(f"[SuryaSetu] Synthetic dataset saved -> {path}  ({len(df)} rows)")
    return path


if __name__ == "__main__":
    save_dataset(Path(__file__).parent)
