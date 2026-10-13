"""
conftest.py — Shared fixtures for SuryaSetu tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src and data directories are importable
SRC_DIR = Path(__file__).parent.parent / "src"
DATA_DIR = Path(__file__).parent.parent / "data"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DATA_DIR))


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Load the real synthetic dataset (generated if missing)."""
    from data_loader import load_data
    return load_data()


@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    """A tiny 48-hour synthetic DataFrame for fast unit tests."""
    from synthetic_data_generator import generate_dataset
    df = generate_dataset()
    df = df.set_index("timestamp")
    return df.head(48)


@pytest.fixture(scope="session")
def forecast_df(raw_df) -> pd.DataFrame:
    """Run the forecast pipeline and return the 7-day forecast."""
    from forecasting import ForecastPipeline
    pipeline = ForecastPipeline(test_days=14)
    pipeline.run(raw_df)
    return pipeline.forecast_df


@pytest.fixture
def default_facilities():
    from essential_services import DEFAULT_FACILITIES
    return list(DEFAULT_FACILITIES)


@pytest.fixture
def shortage_forecast_df(raw_df) -> pd.DataFrame:
    """7-day forecast with solar ×0.1 to guarantee critical shortage."""
    from forecasting import ForecastPipeline
    pipeline = ForecastPipeline(test_days=14)
    pipeline.run(raw_df, solar_scale=0.1)
    return pipeline.forecast_df
