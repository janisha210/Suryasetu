"""
forecasting.py — Solar generation and demand forecasting (Module A).

Pipeline
--------
1. Build features from hourly historical data (no future leakage).
2. Baseline: same-hour-of-day historical mean (rolling 7-day window).
3. ML model: RandomForestRegressor trained on chronological split.
4. Evaluate both on held-out test period (MAE, RMSE).
5. Project 7-day forecast with uncertainty bands (±1 σ from ensemble).

All results are labelled SYNTHETIC where applicable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "day_of_year",
    "cloud_factor",
    "solar_irradiance_factor",
    "solar_lag1h",
    "solar_lag24h",
    "demand_lag1h",
    "demand_lag24h",
    "sin_hour",
    "cos_hour",
    "sin_doy",
    "cos_doy",
]

SOLAR_TARGET = "solar_generation_kw"
DEMAND_TARGET = "total_demand_kw"


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all modelling features. Drops rows with NaN lag values."""
    df = add_cyclic_features(df)
    available = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=available)
    return df


# ---------------------------------------------------------------------------
# Train / test split (chronological — no leakage)
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame, test_days: int = 14
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df chronologically: last *test_days* days become the test set."""
    cutoff = df.index.max() - pd.Timedelta(days=test_days)
    train = df[df.index <= cutoff].copy()
    test = df[df.index > cutoff].copy()
    return train, test


# ---------------------------------------------------------------------------
# Baseline: same-hour historical mean
# ---------------------------------------------------------------------------

class BaselineForecaster:
    """Mean of each hour-of-day computed from training data only."""

    def __init__(self) -> None:
        self._solar_means: dict[int, float] = {}
        self._demand_means: dict[int, float] = {}

    def fit(self, train: pd.DataFrame) -> "BaselineForecaster":
        self._solar_means = (
            train.groupby("hour")[SOLAR_TARGET].mean().to_dict()
        )
        self._demand_means = (
            train.groupby("hour")[DEMAND_TARGET].mean().to_dict()
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        hours = df["hour"] if "hour" in df.columns else df.index.hour
        solar = hours.map(self._solar_means).fillna(0.0)
        demand = hours.map(self._demand_means).fillna(0.0)
        return pd.DataFrame(
            {
                "baseline_solar_kw": solar.values,
                "baseline_demand_kw": demand.values,
            },
            index=df.index,
        )


# ---------------------------------------------------------------------------
# ML model: RandomForest
# ---------------------------------------------------------------------------

@dataclass
class MLForecaster:
    n_estimators: int = 100
    max_depth: int = 12
    random_state: int = 42

    _solar_model: Optional[RandomForestRegressor] = field(default=None, repr=False)
    _demand_model: Optional[RandomForestRegressor] = field(default=None, repr=False)
    _scaler: Optional[StandardScaler] = field(default=None, repr=False)
    _feature_cols: list[str] = field(default_factory=list, repr=False)

    def _build_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(self, train: pd.DataFrame) -> "MLForecaster":
        available = [c for c in FEATURE_COLS if c in train.columns]
        self._feature_cols = available

        X = train[available].values
        y_solar = train[SOLAR_TARGET].values
        y_demand = train[DEMAND_TARGET].values

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._solar_model = self._build_model()
        self._solar_model.fit(X_scaled, y_solar)

        self._demand_model = self._build_model()
        self._demand_model.fit(X_scaled, y_demand)

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._solar_model is None:
            raise RuntimeError("MLForecaster not fitted. Call .fit() first.")

        X = df[self._feature_cols].values
        X_scaled = self._scaler.transform(X)

        # Per-tree predictions for uncertainty estimation
        solar_tree_preds = np.array(
            [t.predict(X_scaled) for t in self._solar_model.estimators_]
        )
        demand_tree_preds = np.array(
            [t.predict(X_scaled) for t in self._demand_model.estimators_]
        )

        ml_solar = np.clip(solar_tree_preds.mean(axis=0), 0, None)
        ml_solar_std = solar_tree_preds.std(axis=0)

        ml_demand = np.clip(demand_tree_preds.mean(axis=0), 0, None)
        ml_demand_std = demand_tree_preds.std(axis=0)

        return pd.DataFrame(
            {
                "ml_solar_kw": ml_solar,
                "ml_solar_std": ml_solar_std,
                "ml_demand_kw": ml_demand,
                "ml_demand_std": ml_demand_std,
            },
            index=df.index,
        )

    @property
    def is_fitted(self) -> bool:
        return self._solar_model is not None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class ForecastMetrics:
    target: str
    model_name: str
    mae: float
    rmse: float
    n_samples: int
    is_synthetic: bool = True

    def as_dict(self) -> dict:
        return {
            "Target": self.target,
            "Model": self.model_name,
            "MAE (kW)": round(self.mae, 4),
            "RMSE (kW)": round(self.rmse, 4),
            "Test samples": self.n_samples,
            "Data": "Synthetic ⚠" if self.is_synthetic else "Real",
        }


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    model_name: str,
    is_synthetic: bool = True,
) -> ForecastMetrics:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    mae = mean_absolute_error(y_t, y_p)
    rmse = mean_squared_error(y_t, y_p) ** 0.5
    return ForecastMetrics(
        target=target,
        model_name=model_name,
        mae=mae,
        rmse=rmse,
        n_samples=int(mask.sum()),
        is_synthetic=is_synthetic,
    )


# ---------------------------------------------------------------------------
# 7-day forecast generator
# ---------------------------------------------------------------------------

def make_7day_forecast(
    df: pd.DataFrame,
    baseline: BaselineForecaster,
    ml_model: Optional[MLForecaster],
    solar_scale: float = 1.0,
    demand_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Generate a 7-day hourly forecast starting from the last timestamp + 1 h.

    Strategy:
    - Repeat the historical same-hour-of-day mean (baseline).
    - If ML model is fitted, use it on synthetic feature rows.
    - Apply solar_scale and demand_scale for scenario testing.
    - Uncertainty band: ±1 σ from ML ensemble (if available), else ±10% baseline.
    """
    last_ts = df.index.max()
    future_index = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=7 * 24, freq="h")

    future_df = pd.DataFrame(index=future_index)
    future_df["hour"] = future_df.index.hour
    future_df["day_of_week"] = future_df.index.dayofweek
    future_df["day_of_year"] = future_df.index.dayofyear

    # Cloud factor: use mean of last 7 days as a proxy
    cutoff_7d = df.index.max() - pd.Timedelta(days=7)
    recent = df[df.index > cutoff_7d]
    cf_by_doy = recent.groupby("day_of_year")["cloud_factor"].mean()
    future_df["cloud_factor"] = [
        cf_by_doy.get(d, float(recent["cloud_factor"].mean()))
        for d in future_df["day_of_year"]
    ]

    # Irradiance factor
    from data.synthetic_data_generator import solar_profile
    future_df["solar_irradiance_factor"] = [
        solar_profile(r.hour, r.day_of_year) * future_df.at[ts, "cloud_factor"]
        for ts, r in future_df.iterrows()
    ]

    # Lag features: use last known actuals and roll forward
    future_df["solar_lag1h"] = np.nan
    future_df["solar_lag24h"] = np.nan
    future_df["demand_lag1h"] = np.nan
    future_df["demand_lag24h"] = np.nan

    # Seed lags from history
    combined_solar = df[SOLAR_TARGET].copy()
    combined_demand = df[DEMAND_TARGET].copy()

    # Baseline predictions for future
    baseline_preds = baseline.predict(future_df)
    solar_base = baseline_preds["baseline_solar_kw"].values
    demand_base = baseline_preds["baseline_demand_kw"].values

    # ML predictions (if available)
    ml_solar = np.full(len(future_df), np.nan)
    ml_solar_std = np.full(len(future_df), np.nan)
    ml_demand = np.full(len(future_df), np.nan)
    ml_demand_std = np.full(len(future_df), np.nan)

    if ml_model is not None and ml_model.is_fitted:
        # Populate lags iteratively
        hist_solar = list(df[SOLAR_TARGET].iloc[-24:])
        hist_demand = list(df[DEMAND_TARGET].iloc[-24:])

        future_rows = []
        for i, (ts, row) in enumerate(future_df.iterrows()):
            row_dict = row.to_dict()
            row_dict["solar_lag1h"] = hist_solar[-1] if hist_solar else 0.0
            row_dict["demand_lag1h"] = hist_demand[-1] if hist_demand else 0.0
            row_dict["solar_lag24h"] = hist_solar[-24] if len(hist_solar) >= 24 else solar_base[i]
            row_dict["demand_lag24h"] = hist_demand[-24] if len(hist_demand) >= 24 else demand_base[i]
            future_rows.append(row_dict)

            # Add cyclic features
            row_dict["sin_hour"] = np.sin(2 * np.pi * row["hour"] / 24)
            row_dict["cos_hour"] = np.cos(2 * np.pi * row["hour"] / 24)
            row_dict["sin_doy"] = np.sin(2 * np.pi * row["day_of_year"] / 365)
            row_dict["cos_doy"] = np.cos(2 * np.pi * row["day_of_year"] / 365)

            # Predict for this row (use baseline if model fails)
            try:
                row_df = pd.DataFrame([row_dict], index=[ts])
                preds = ml_model.predict(row_df)
                s = float(preds["ml_solar_kw"].iloc[0])
                d = float(preds["ml_demand_kw"].iloc[0])
            except Exception:
                s = solar_base[i]
                d = demand_base[i]

            ml_solar[i] = max(0.0, s)
            ml_demand[i] = max(0.0, d)
            hist_solar.append(s)
            hist_demand.append(d)

        # std from static ensemble: recompute on full batch
        try:
            feat_rows = pd.DataFrame(future_rows, index=future_index)
            for col in ["sin_hour", "cos_hour", "sin_doy", "cos_doy"]:
                if col not in feat_rows.columns:
                    feat_rows[col] = feat_rows["hour"].apply(
                        lambda h: np.sin(2*np.pi*h/24) if "sin" in col else np.cos(2*np.pi*h/24)
                    )
            preds_batch = ml_model.predict(feat_rows)
            ml_solar_std = preds_batch["ml_solar_std"].values
            ml_demand_std = preds_batch["ml_demand_std"].values
        except Exception:
            ml_solar_std = solar_base * 0.10
            ml_demand_std = demand_base * 0.10
    else:
        ml_solar_std = solar_base * 0.10
        ml_demand_std = demand_base * 0.10

    # Choose primary forecast
    primary_solar = np.where(~np.isnan(ml_solar), ml_solar, solar_base)
    primary_demand = np.where(~np.isnan(ml_demand), ml_demand, demand_base)

    # Apply scenario scales
    primary_solar = primary_solar * solar_scale
    primary_demand = primary_demand * demand_scale
    solar_base_scaled = solar_base * solar_scale
    demand_base_scaled = demand_base * demand_scale

    result = pd.DataFrame(
        {
            "forecast_solar_kw": np.clip(primary_solar, 0, None),
            "forecast_solar_low_kw": np.clip(primary_solar - ml_solar_std * solar_scale, 0, None),
            "forecast_solar_high_kw": np.clip(primary_solar + ml_solar_std * solar_scale, 0, None),
            "forecast_demand_kw": np.clip(primary_demand, 0, None),
            "forecast_demand_low_kw": np.clip(primary_demand - ml_demand_std * demand_scale, 0, None),
            "forecast_demand_high_kw": np.clip(primary_demand + ml_demand_std * demand_scale, 0, None),
            "baseline_solar_kw": np.clip(solar_base_scaled, 0, None),
            "baseline_demand_kw": np.clip(demand_base_scaled, 0, None),
            "net_balance_kw": primary_solar - primary_demand,
            "is_synthetic": True,
        },
        index=future_index,
    )
    result.index.name = "timestamp"
    return result


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

@dataclass
class ForecastPipeline:
    """Orchestrates baseline + ML training, evaluation and forecasting."""
    test_days: int = 14
    is_synthetic: bool = True

    baseline: BaselineForecaster = field(default_factory=BaselineForecaster)
    ml_model: MLForecaster = field(default_factory=MLForecaster)

    metrics: list[ForecastMetrics] = field(default_factory=list)
    forecast_df: Optional[pd.DataFrame] = field(default=None, repr=False)

    def run(
        self,
        df: pd.DataFrame,
        solar_scale: float = 1.0,
        demand_scale: float = 1.0,
    ) -> "ForecastPipeline":
        feat_df = build_features(df)

        if len(feat_df) < 48:
            raise ValueError(
                "Too few valid feature rows to run forecasting. "
                "Ensure the dataset has at least 48 hours of data."
            )

        train, test = chronological_split(feat_df, test_days=self.test_days)

        # --- Baseline ---
        self.baseline.fit(train)

        # --- ML ---
        self.ml_model.fit(train)

        # --- Evaluate on test set ---
        if len(test) >= 24:
            base_preds = self.baseline.predict(test)
            ml_preds = self.ml_model.predict(test)

            for target_col, base_col, ml_col in [
                (SOLAR_TARGET, "baseline_solar_kw", "ml_solar_kw"),
                (DEMAND_TARGET, "baseline_demand_kw", "ml_demand_kw"),
            ]:
                y_true = test[target_col].values
                self.metrics.append(
                    evaluate_forecast(
                        y_true, base_preds[base_col].values,
                        target=target_col, model_name="Baseline (hour-mean)",
                        is_synthetic=self.is_synthetic,
                    )
                )
                self.metrics.append(
                    evaluate_forecast(
                        y_true, ml_preds[ml_col].values,
                        target=target_col, model_name="RandomForest",
                        is_synthetic=self.is_synthetic,
                    )
                )

        # --- 7-day forecast ---
        self.forecast_df = make_7day_forecast(
            df, self.baseline, self.ml_model,
            solar_scale=solar_scale, demand_scale=demand_scale,
        )
        return self

    def get_metrics_df(self) -> pd.DataFrame:
        if not self.metrics:
            return pd.DataFrame()
        return pd.DataFrame([m.as_dict() for m in self.metrics])
