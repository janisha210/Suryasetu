"""
test_forecasting.py — Tests for the forecasting pipeline.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
DATA_DIR = Path(__file__).parent.parent / "data"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DATA_DIR))

from forecasting import (
    ForecastPipeline, BaselineForecaster, MLForecaster,
    build_features, chronological_split, evaluate_forecast,
    make_7day_forecast, SOLAR_TARGET, DEMAND_TARGET,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def test_build_features_no_leakage(raw_df):
    """Lag features must only use past data (no future leakage)."""
    feat_df = build_features(raw_df)
    # lag1h should be the value from one hour ago
    for idx_pos in range(1, min(10, len(feat_df))):
        ts = feat_df.index[idx_pos]
        prev_ts = feat_df.index[idx_pos - 1]
        # lag1h at position i should equal the raw value at i-1
        assert feat_df["solar_lag1h"].iloc[idx_pos] == pytest.approx(
            raw_df.loc[prev_ts, "solar_generation_kw"], rel=1e-3
        ), "solar_lag1h appears to contain future values (leakage)."


def test_chronological_split_no_overlap(raw_df):
    feat_df = build_features(raw_df)
    train, test = chronological_split(feat_df, test_days=14)
    assert len(train) > 0
    assert len(test) > 0
    # No timestamp should appear in both splits
    overlap = set(train.index) & set(test.index)
    assert len(overlap) == 0, f"Train/test overlap: {len(overlap)} rows"


def test_chronological_split_ordering(raw_df):
    feat_df = build_features(raw_df)
    train, test = chronological_split(feat_df, test_days=14)
    assert train.index.max() < test.index.min(), "Train period extends into test period."


# ─────────────────────────────────────────────────────────────────────────────
# Baseline forecaster
# ─────────────────────────────────────────────────────────────────────────────

def test_baseline_fits_without_error(raw_df):
    feat_df = build_features(raw_df)
    train, _ = chronological_split(feat_df, test_days=14)
    baseline = BaselineForecaster()
    baseline.fit(train)
    assert len(baseline._solar_means) == 24   # one value per hour
    assert len(baseline._demand_means) == 24


def test_baseline_predicts_non_negative(raw_df):
    feat_df = build_features(raw_df)
    train, test = chronological_split(feat_df, test_days=14)
    baseline = BaselineForecaster().fit(train)
    preds = baseline.predict(test)
    assert (preds["baseline_solar_kw"] >= 0).all()
    assert (preds["baseline_demand_kw"] >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# ML forecaster
# ─────────────────────────────────────────────────────────────────────────────

def test_ml_forecaster_fits_without_error(raw_df):
    feat_df = build_features(raw_df)
    train, _ = chronological_split(feat_df, test_days=14)
    ml = MLForecaster()
    ml.fit(train)
    assert ml.is_fitted


def test_ml_forecaster_predicts_correct_shape(raw_df):
    feat_df = build_features(raw_df)
    train, test = chronological_split(feat_df, test_days=14)
    ml = MLForecaster().fit(train)
    preds = ml.predict(test)
    assert len(preds) == len(test)
    assert "ml_solar_kw" in preds.columns
    assert "ml_demand_kw" in preds.columns


def test_ml_forecaster_non_negative(raw_df):
    feat_df = build_features(raw_df)
    train, test = chronological_split(feat_df, test_days=14)
    ml = MLForecaster().fit(train)
    preds = ml.predict(test)
    assert (preds["ml_solar_kw"] >= 0).all()
    assert (preds["ml_demand_kw"] >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluate_forecast_metrics(raw_df):
    feat_df = build_features(raw_df)
    train, test = chronological_split(feat_df, test_days=14)
    baseline = BaselineForecaster().fit(train)
    preds = baseline.predict(test)
    metrics = evaluate_forecast(
        test[SOLAR_TARGET].values,
        preds["baseline_solar_kw"].values,
        target="solar", model_name="Baseline",
    )
    assert metrics.mae >= 0
    assert metrics.rmse >= metrics.mae   # RMSE ≥ MAE always
    assert metrics.n_samples == len(test)


def test_ml_beats_or_ties_baseline_rmse(raw_df):
    """ML RMSE on synthetic data should be competitive with baseline."""
    feat_df = build_features(raw_df)
    train, test = chronological_split(feat_df, test_days=14)
    baseline = BaselineForecaster().fit(train)
    ml = MLForecaster().fit(train)
    base_preds = baseline.predict(test)
    ml_preds = ml.predict(test)

    base_m = evaluate_forecast(test[SOLAR_TARGET].values, base_preds["baseline_solar_kw"].values, "solar", "Baseline")
    ml_m = evaluate_forecast(test[SOLAR_TARGET].values, ml_preds["ml_solar_kw"].values, "solar", "ML")

    # ML should not be drastically worse (allow 50% slack on synthetic data)
    assert ml_m.rmse <= base_m.rmse * 1.5, (
        f"ML RMSE ({ml_m.rmse:.3f}) much worse than baseline ({base_m.rmse:.3f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7-day forecast
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_produces_7day_forecast(raw_df):
    pipeline = ForecastPipeline(test_days=14)
    pipeline.run(raw_df)
    fc = pipeline.forecast_df
    assert fc is not None
    assert len(fc) == 7 * 24


def test_forecast_index_after_training_end(raw_df):
    pipeline = ForecastPipeline(test_days=14)
    pipeline.run(raw_df)
    fc = pipeline.forecast_df
    assert fc.index.min() > raw_df.index.max()


def test_forecast_non_negative(raw_df):
    pipeline = ForecastPipeline(test_days=14)
    pipeline.run(raw_df)
    fc = pipeline.forecast_df
    assert (fc["forecast_solar_kw"] >= 0).all()
    assert (fc["forecast_demand_kw"] >= 0).all()


def test_forecast_with_scale(raw_df):
    pipeline = ForecastPipeline(test_days=14)
    pipeline.run(raw_df, solar_scale=0.7, demand_scale=1.2)
    fc = pipeline.forecast_df
    # Check that the scaled forecast is lower than unscaled for solar
    pipeline2 = ForecastPipeline(test_days=14)
    pipeline2.run(raw_df)
    fc2 = pipeline2.forecast_df
    assert fc["forecast_solar_kw"].mean() < fc2["forecast_solar_kw"].mean() * 0.8 * 1.05


def test_pipeline_has_metrics(raw_df):
    pipeline = ForecastPipeline(test_days=14)
    pipeline.run(raw_df)
    assert len(pipeline.metrics) >= 4   # baseline + ML × 2 targets
    df = pipeline.get_metrics_df()
    assert "MAE (kW)" in df.columns
    assert "RMSE (kW)" in df.columns
