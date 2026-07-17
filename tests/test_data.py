"""Tests for the data layer."""

import pandas as pd
import pytest

from src.data.base import validate
from src.data.synthetic import SyntheticDataProvider


def test_synthetic_provider_returns_standard_schema():
    df = SyntheticDataProvider(n_periods=100, seed=1).fetch()

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "timestamp"
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 100
    assert df.index.is_monotonic_increasing
    assert (df.dtypes == float).all()


def test_synthetic_provider_is_reproducible_with_seed():
    df1 = SyntheticDataProvider(n_periods=50, seed=7).fetch()
    df2 = SyntheticDataProvider(n_periods=50, seed=7).fetch()

    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_provider_different_seeds_differ():
    df1 = SyntheticDataProvider(n_periods=50, seed=1).fetch()
    df2 = SyntheticDataProvider(n_periods=50, seed=2).fetch()

    assert not df1["close"].equals(df2["close"])


def test_validate_ohlcv_schema_rejects_non_datetime_index():
    df = pd.DataFrame(
        {"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}
    )
    with pytest.raises(TypeError):
        validate(df)


def test_validate_ohlcv_schema_rejects_wrong_index_name():
    index = pd.date_range("2022-01-01", periods=2, name="date")
    df = pd.DataFrame(
        {"open": [1, 1], "high": [1, 1], "low": [1, 1], "close": [1, 1], "volume": [1, 1]},
        index=index,
    )
    with pytest.raises(ValueError):
        validate(df)


def test_validate_ohlcv_schema_rejects_missing_columns():
    index = pd.date_range("2022-01-01", periods=2, name="timestamp")
    df = pd.DataFrame({"open": [1, 1], "close": [1, 1]}, index=index)
    with pytest.raises(ValueError):
        validate(df)
