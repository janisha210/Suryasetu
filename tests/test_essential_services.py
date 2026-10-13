"""
test_essential_services.py — Tests for essential-services protection layer.
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

from essential_services import (
    evaluate_service_coverage, DEFAULT_FACILITIES, Facility,
    estimate_water_volume, facility_from_dict,
)


# ─────────────────────────────────────────────────────────────────────────────
# Coverage with adequate supply
# ─────────────────────────────────────────────────────────────────────────────

def test_no_shortage_with_ample_solar(forecast_df, default_facilities):
    """With solar ×2, critical loads should be comfortably covered."""
    report = evaluate_service_coverage(
        forecast_df, default_facilities, solar_scale=2.0
    )
    assert not report.has_critical_shortage, (
        "Expected no critical shortage with solar ×2.0"
    )


def test_coverage_always_reported(forecast_df, default_facilities):
    """daily_coverage must have rows regardless of shortage status."""
    report = evaluate_service_coverage(
        forecast_df, default_facilities, solar_scale=1.0
    )
    assert len(report.daily_coverage) > 0


def test_coverage_fractions_0_to_100(forecast_df, default_facilities):
    report = evaluate_service_coverage(forecast_df, default_facilities)
    for cov in report.daily_coverage:
        assert 0.0 <= cov.coverage_pct <= 100.0 + 1e-6, (
            f"Coverage {cov.coverage_pct}% out of [0, 100] for {cov.facility_label}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Shortage scenario — critical demand must exceed supply
# ─────────────────────────────────────────────────────────────────────────────

def test_critical_shortage_is_visible(shortage_forecast_df, default_facilities):
    """
    With solar ×0.1 (10% of normal), critical loads MUST be under-served.
    The report must report this explicitly — never silently hide it.
    """
    report = evaluate_service_coverage(
        shortage_forecast_df, default_facilities, solar_scale=0.1
    )
    assert report.has_critical_shortage, (
        "FAILURE: Critical shortage not reported when solar supply is only 10% "
        "of normal demand. The system must never silently hide shortages."
    )
    # Affected facilities should be in the report
    shortage_labels = {cs.facility_label for cs in report.critical_shortages}
    assert len(shortage_labels) >= 1


def test_shortage_kwh_positive(shortage_forecast_df, default_facilities):
    report = evaluate_service_coverage(
        shortage_forecast_df, default_facilities, solar_scale=0.1
    )
    for cs in report.critical_shortages:
        assert cs.shortage_kwh > 0.0, "Shortage kWh should be positive when shortage reported."


def test_shortage_warning_messages_not_empty(shortage_forecast_df, default_facilities):
    report = evaluate_service_coverage(
        shortage_forecast_df, default_facilities, solar_scale=0.1
    )
    real_warnings = [w for w in report.warnings if w.strip()]
    assert len(real_warnings) > 0, "Warning messages must be generated for critical shortages."


# ─────────────────────────────────────────────────────────────────────────────
# Priority ordering
# ─────────────────────────────────────────────────────────────────────────────

def test_critical_served_before_important(forecast_df, default_facilities):
    """
    With limited supply (solar ×0.5), check that critical facilities
    have ≥ their minimum before important/flexible.
    """
    report = evaluate_service_coverage(
        forecast_df, default_facilities, solar_scale=0.5
    )
    cov_by_priority = {}
    for cov in report.daily_coverage:
        p = cov.priority
        cov_by_priority.setdefault(p, []).append(cov.coverage_pct)

    # Average critical coverage must be ≥ average flexible coverage
    if "critical" in cov_by_priority and "flexible" in cov_by_priority:
        avg_crit = sum(cov_by_priority["critical"]) / len(cov_by_priority["critical"])
        avg_flex = sum(cov_by_priority["flexible"]) / len(cov_by_priority["flexible"])
        assert avg_crit >= avg_flex - 5.0, (
            f"Critical avg coverage {avg_crit:.1f}% < flexible {avg_flex:.1f}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Water volume helper
# ─────────────────────────────────────────────────────────────────────────────

def test_water_volume_positive():
    litres = estimate_water_volume(energy_kwh=10.0, pump_efficiency_kwh_per_kl=0.5)
    assert litres > 0


def test_water_volume_units():
    # 10 kWh ÷ 0.5 kWh/kL = 20 kL = 20 000 L
    litres = estimate_water_volume(10.0, 0.5)
    assert abs(litres - 20_000) < 1.0


def test_water_volume_invalid_efficiency():
    with pytest.raises(ValueError):
        estimate_water_volume(10.0, pump_efficiency_kwh_per_kl=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Facility from dict
# ─────────────────────────────────────────────────────────────────────────────

def test_facility_from_dict():
    d = {
        "key": "test_fac",
        "label": "Test",
        "priority": "important",
        "min_energy_kwh_day": 8.0,
        "op_start": 7,
        "op_end": 17,
        "peak_demand_kw": 3.0,
        "is_flexible": True,
        "sched_start": 8,
        "sched_end": 16,
    }
    fac = facility_from_dict(d)
    assert fac.key == "test_fac"
    assert fac.priority == "important"
    assert fac.operating_hours == (7, 17)
    assert fac.scheduling_window == (8, 16)
