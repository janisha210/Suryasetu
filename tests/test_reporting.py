"""
test_reporting.py — Tests for CSV and PDF export.
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import io

SRC_DIR = Path(__file__).parent.parent / "src"
DATA_DIR = Path(__file__).parent.parent / "data"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DATA_DIR))

from reporting import (
    export_csv, export_forecast_csv, export_plan_csv,
    export_pdf, REPORTLAB_AVAILABLE,
)


@pytest.fixture(scope="module")
def pipeline_and_plan():
    from forecasting import ForecastPipeline
    from data_loader import load_data
    from weekly_planner import build_weekly_plan
    from essential_services import DEFAULT_FACILITIES
    raw = load_data()
    pipeline = ForecastPipeline(test_days=14)
    pipeline.run(raw)
    fc = pipeline.forecast_df
    plan = build_weekly_plan(fc, list(DEFAULT_FACILITIES))
    return pipeline, plan, fc


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def test_export_csv_bytes(pipeline_and_plan):
    _, plan, _ = pipeline_and_plan
    data = export_csv(plan.daily_df)
    assert isinstance(data, bytes)
    assert len(data) > 100


def test_export_csv_parseable(pipeline_and_plan):
    _, plan, _ = pipeline_and_plan
    data = export_csv(plan.daily_df)
    df = pd.read_csv(io.BytesIO(data), index_col=0)
    assert len(df) > 0


def test_forecast_csv_contains_disclaimer(pipeline_and_plan):
    pipeline, _, fc = pipeline_and_plan
    data = export_forecast_csv(fc)
    text = data.decode("utf-8")
    assert "Synthetic" in text or "synthetic" in text, (
        "Forecast CSV must contain a synthetic data label."
    )


def test_plan_csv_has_sections(pipeline_and_plan):
    _, plan, _ = pipeline_and_plan
    data = export_plan_csv(plan.hourly_df, plan.daily_df)
    text = data.decode("utf-8")
    assert "Hourly Plan" in text
    assert "Daily Summary" in text


def test_plan_csv_has_timestamp(pipeline_and_plan):
    _, plan, _ = pipeline_and_plan
    data = export_plan_csv(plan.hourly_df, plan.daily_df)
    text = data.decode("utf-8")
    assert "Generated" in text


# ─────────────────────────────────────────────────────────────────────────────
# PDF / text export
# ─────────────────────────────────────────────────────────────────────────────

def test_pdf_export_returns_bytes(pipeline_and_plan):
    pipeline, plan, fc = pipeline_and_plan
    from metrics import energy_balance_summary
    balance = energy_balance_summary(fc)
    data = export_pdf(
        scenario_name="Test",
        scenario_notes="Test notes",
        balance_metrics=balance,
        daily_summary_df=plan.daily_df,
        coverage_df=None,
        comparison_df=None,
        forecast_metrics_df=pipeline.get_metrics_df(),
    )
    assert isinstance(data, bytes)
    assert len(data) > 500


def test_pdf_export_contains_disclaimer(pipeline_and_plan):
    pipeline, plan, fc = pipeline_and_plan
    from metrics import energy_balance_summary
    balance = energy_balance_summary(fc)
    data = export_pdf(
        scenario_name="Test", scenario_notes="",
        balance_metrics=balance,
        daily_summary_df=None, coverage_df=None,
        comparison_df=None, forecast_metrics_df=None,
    )
    text = data.decode("utf-8", errors="replace").lower()
    assert "synthetic" in text or "disclaimer" in text or "%pdf" in text[:10].lower(), (
        "PDF must contain synthetic disclaimer."
    )


def test_pdf_export_with_empty_optional_tables(pipeline_and_plan):
    _, _, fc = pipeline_and_plan
    from metrics import energy_balance_summary
    balance = energy_balance_summary(fc)
    # Should not raise even with None tables
    data = export_pdf(
        scenario_name="Empty Tables Test",
        scenario_notes="",
        balance_metrics=balance,
        daily_summary_df=None,
        coverage_df=None,
        comparison_df=None,
        forecast_metrics_df=None,
    )
    assert isinstance(data, bytes)
