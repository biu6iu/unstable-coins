"""Tests for the strategy layer."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.buy_and_hold import BuyAndHoldStrategy
from src.strategies.ma_crossover import MACrossoverStrategy


def _make_ohlcv(close):
    close = np.asarray(close, dtype=float)
    index = pd.date_range("2022-01-01", periods=len(close), freq="D", name="timestamp")
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(len(close), 100.0),
        },
        index=index,
    )


def test_ma_crossover_flips_signal_on_hand_computed_bars():
    # Hand-verified with fast=2/slow=3 SMAs:
    #   sma_2: [nan, 10, 10, 11, 13, 15, 13, 9, 7, 5]
    #   sma_3: [nan, nan, 10, 10.667, 12, 14, 13.333, 11.333, 8, 6]
    # sma_2 crosses above sma_3 at bar 3, back below at bar 6.
    close = [10, 10, 10, 12, 14, 16, 10, 8, 6, 4]
    df = _make_ohlcv(close)

    out = MACrossoverStrategy(fast=2, slow=3).generate_signals(df)

    expected_signal = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert list(out["signal"]) == expected_signal


def test_ma_crossover_rejects_fast_greater_or_equal_slow():
    with pytest.raises(ValueError):
        MACrossoverStrategy(fast=50, slow=20)
    with pytest.raises(ValueError):
        MACrossoverStrategy(fast=20, slow=20)


def test_ma_crossover_does_not_add_position_or_mutate_input():
    close = [10, 11, 12, 13, 14]
    df = _make_ohlcv(close)
    original = df.copy()

    out = MACrossoverStrategy(fast=2, slow=3).generate_signals(df)

    pd.testing.assert_frame_equal(df, original)
    assert "position" not in out.columns
    assert "signal" in out.columns


def test_buy_and_hold_is_always_long():
    close = [10, 9, 11, 8, 20]
    df = _make_ohlcv(close)

    out = BuyAndHoldStrategy().generate_signals(df)

    assert (out["signal"] == 1).all()


def test_buy_and_hold_does_not_mutate_input():
    close = [10, 9, 11]
    df = _make_ohlcv(close)
    original = df.copy()

    BuyAndHoldStrategy().generate_signals(df)

    pd.testing.assert_frame_equal(df, original)


def test_strategy_names():
    assert MACrossoverStrategy(fast=20, slow=50).name == "MACrossover(20,50)"
    assert BuyAndHoldStrategy().name == "BuyAndHold"
