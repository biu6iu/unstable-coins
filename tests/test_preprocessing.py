"""Tests for preprocessing: DataCleaner and FeatureEngineer."""

import logging

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.features import FeatureEngineer


def _make_ohlcv(index, close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(close), 100.0),
        },
        index=index,
    )


# ---------- DataCleaner ----------


def test_cleaner_drops_duplicate_timestamps(caplog):
    index = pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-02", "2022-01-03"])
    index.name = "timestamp"
    df = _make_ohlcv(index, [1, 2, 2, 3])

    with caplog.at_level(logging.WARNING):
        cleaned = DataCleaner().clean(df)

    assert len(cleaned) == 3
    assert cleaned.index.is_unique
    assert any("duplicate" in message.lower() for message in caplog.messages)


def test_cleaner_sorts_index():
    index = pd.to_datetime(["2022-01-03", "2022-01-01", "2022-01-02"])
    index.name = "timestamp"
    df = _make_ohlcv(index, [3, 1, 2])

    cleaned = DataCleaner().clean(df)

    assert cleaned.index.is_monotonic_increasing
    assert list(cleaned["close"]) == [1, 2, 3]


def test_cleaner_forward_fills_small_gaps(caplog):
    index = pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-04", "2022-01-05"])
    index.name = "timestamp"
    df = _make_ohlcv(index, [1, 2, 4, 5])

    with caplog.at_level(logging.WARNING):
        cleaned = DataCleaner(max_ffill=3).clean(df)

    assert pd.Timestamp("2022-01-03") in cleaned.index
    assert cleaned.loc["2022-01-03", "close"] == 2  # forward-filled from Jan 2
    assert any("fill" in message.lower() for message in caplog.messages)


def test_cleaner_enforces_float_dtype():
    index = pd.to_datetime(["2022-01-01", "2022-01-02"])
    index.name = "timestamp"
    df = pd.DataFrame(
        {
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
            "volume": [100, 100],
        },
        index=index,
    )

    cleaned = DataCleaner().clean(df)

    assert (cleaned.dtypes == float).all()


# ---------- FeatureEngineer ----------


@pytest.fixture
def engineer():
    return FeatureEngineer()


def _sample_df():
    index = pd.date_range("2022-01-01", periods=10, freq="D", name="timestamp")
    close = [10, 11, 12, 11, 10, 9, 10, 11, 12, 13]
    return _make_ohlcv(index, close)


def test_sma_matches_hand_computed_value(engineer):
    df = _sample_df()
    out = engineer.sma(df, window=3)

    expected_last = sum(df["close"].iloc[-3:]) / 3
    assert out["sma_3"].iloc[-1] == pytest.approx(expected_last)
    assert out["sma_3"].iloc[:2].isna().all()


def test_ema_does_not_mutate_input(engineer):
    df = _sample_df()
    original = df.copy()

    out = engineer.ema(df, window=3)

    pd.testing.assert_frame_equal(df, original)
    assert "ema_3" in out.columns
    assert "ema_3" not in df.columns


def test_returns_matches_hand_computed_value(engineer):
    df = _sample_df()
    out = engineer.returns(df)

    expected = (df["close"].iloc[1] - df["close"].iloc[0]) / df["close"].iloc[0]
    assert out["returns"].iloc[1] == pytest.approx(expected)
    assert pd.isna(out["returns"].iloc[0])


def test_volatility_is_rolling_std_of_returns(engineer):
    df = _sample_df()
    out = engineer.volatility(df, window=3)

    rets = df["close"].pct_change()
    expected = rets.iloc[1:4].std()
    assert out["volatility_3"].iloc[3] == pytest.approx(expected)


def test_rsi_is_bounded_between_0_and_100(engineer):
    df = _sample_df()
    out = engineer.rsi(df, window=3)

    valid = out["rsi_3"].dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


@pytest.mark.parametrize(
    "method_name, kwargs, column",
    [
        ("sma", {"window": 3}, "sma_3"),
        ("ema", {"window": 3}, "ema_3"),
        ("rsi", {"window": 3}, "rsi_3"),
        ("returns", {}, "returns"),
        ("volatility", {"window": 3}, "volatility_3"),
    ],
)
def test_feature_values_are_lookahead_free(engineer, method_name, kwargs, column):
    """Feature values at time t must not change when future rows are appended."""
    df = _sample_df()
    cutoff = 6

    before = getattr(engineer, method_name)(df.iloc[:cutoff], **kwargs)
    after = getattr(engineer, method_name)(df, **kwargs)

    pd.testing.assert_series_equal(
        before[column], after[column].iloc[:cutoff], check_names=False
    )
