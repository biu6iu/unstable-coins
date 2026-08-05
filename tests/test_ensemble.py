import numpy as np
import pandas as pd
import pytest

from src.backtest.result import BacktestResult
from src.strategies.base import Strategy
from src.strategies.ensemble import (
    SoftVotingStrategy,
    VotingStrategy,
    correlation_matrix,
)
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.rsi_filter import RSIFilterStrategy
from src.strategies.tsmom import TimeSeriesMomentumStrategy


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


class _FixedSignalStrategy(Strategy):
    """Test double: emits a caller-supplied signal instead of computing one."""

    def __init__(self, signal, name="Fixed"):
        self._signal = signal
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = self._signal
        return out


def _make_result(strategy_name, returns_values):
    index = pd.date_range("2022-01-01", periods=len(returns_values), freq="D", name="timestamp")
    returns = pd.Series(returns_values, index=index, dtype=float)
    equity = 1000.0 * (1 + returns).cumprod()
    zeros = pd.Series(0.0, index=index)
    df = pd.DataFrame({"close": equity}, index=index)
    return BacktestResult(
        df=df,
        positions=pd.Series(1, index=index),
        strategy_returns=returns,
        equity_curve=equity,
        trade_count=1,
        strategy_name=strategy_name,
        gross_returns=returns,
        fee_drag=zeros,
        slippage_drag=zeros,
        cash=zeros,
        units=zeros,
    )


# Three members with vote counts of 3, 2, 1, 1, 1 across the five bars, so
# k=1/2/3 each produce a different signal series.
_M1 = [1, 1, 1, 0, 0]
_M2 = [1, 1, 0, 1, 0]
_M3 = [1, 0, 0, 0, 1]


def _members():
    return [
        _FixedSignalStrategy(_M1, name="M1"),
        _FixedSignalStrategy(_M2, name="M2"),
        _FixedSignalStrategy(_M3, name="M3"),
    ]


def test_voting_strategy_k_equals_1_is_logical_or():
    df = _make_ohlcv([10, 11, 12, 13, 14])
    out = VotingStrategy(_members(), k=1).generate_signals(df)
    assert list(out["signal"]) == [1, 1, 1, 1, 1]


def test_voting_strategy_k_equals_2_requires_majority():
    df = _make_ohlcv([10, 11, 12, 13, 14])
    out = VotingStrategy(_members(), k=2).generate_signals(df)
    assert list(out["signal"]) == [1, 1, 0, 0, 0]


def test_voting_strategy_k_equals_3_is_logical_and():
    df = _make_ohlcv([10, 11, 12, 13, 14])
    out = VotingStrategy(_members(), k=3).generate_signals(df)
    assert list(out["signal"]) == [1, 0, 0, 0, 0]


def test_voting_strategy_rejects_k_out_of_range():
    with pytest.raises(ValueError):
        VotingStrategy(_members(), k=0)
    with pytest.raises(ValueError):
        VotingStrategy(_members(), k=4)


def test_voting_strategy_does_not_mutate_input():
    df = _make_ohlcv([10, 11, 12, 13, 14])
    original = df.copy()

    VotingStrategy(_members(), k=2).generate_signals(df)

    pd.testing.assert_frame_equal(df, original)


def test_voting_strategy_name_includes_members_and_k():
    ensemble = VotingStrategy(_members(), k=2)
    assert ensemble.name == "Voting(M1,M2,M3;k=2)"


def test_voting_strategy_is_nestable_with_rsi_filter_member():
    # An RSI-filtered crossover as one member among others must work
    # unchanged - the ensemble only cares that members implement Strategy.
    close = [10 + i for i in range(30)]
    df = _make_ohlcv(close)
    members = [
        RSIFilterStrategy(MACrossoverStrategy(fast=2, slow=3), rsi_overbought=70.0),
        TimeSeriesMomentumStrategy(lookback=2),
    ]

    out = VotingStrategy(members, k=1).generate_signals(df)

    assert set(out["signal"].unique()) <= {0, 1}
    assert len(out) == len(df)


def test_soft_voting_unweighted_is_mean_of_members():
    df = _make_ohlcv([10, 11, 12])
    members = [
        _FixedSignalStrategy([1, 0, 1], name="A"),
        _FixedSignalStrategy([0, 1, 1], name="B"),
    ]

    out = SoftVotingStrategy(members).generate_signals(df)

    assert list(out["signal"]) == pytest.approx([0.5, 0.5, 1.0])


def test_soft_voting_weighted_average():
    df = _make_ohlcv([10, 11, 12])
    members = [
        _FixedSignalStrategy([1, 0, 1], name="A"),
        _FixedSignalStrategy([0, 0, 1], name="B"),
        _FixedSignalStrategy([1, 1, 0], name="C"),
    ]

    out = SoftVotingStrategy(members, weights=[1.0, 2.0, 1.0]).generate_signals(df)

    assert list(out["signal"]) == pytest.approx([0.5, 0.25, 0.75])


def test_soft_voting_rejects_mismatched_weights_length():
    with pytest.raises(ValueError):
        SoftVotingStrategy(_members(), weights=[1.0, 2.0])


def test_soft_voting_does_not_mutate_input():
    df = _make_ohlcv([10, 11, 12])
    original = df.copy()
    members = [_FixedSignalStrategy([1, 0, 1], name="A"), _FixedSignalStrategy([0, 1, 1], name="B")]

    SoftVotingStrategy(members).generate_signals(df)

    pd.testing.assert_frame_equal(df, original)


def test_soft_voting_name_includes_members():
    ensemble = SoftVotingStrategy([_FixedSignalStrategy([1], name="A"), _FixedSignalStrategy([0], name="B")])
    assert ensemble.name == "SoftVoting(A,B)"


def test_correlation_matrix_identifies_perfectly_correlated_members():
    identical_returns = [0.01, -0.02, 0.03, 0.0, -0.01]
    results = [
        _make_result("A", identical_returns),
        _make_result("B", identical_returns),
    ]

    matrix = correlation_matrix(results)

    assert matrix.loc["A", "B"] == pytest.approx(1.0)
    assert matrix.loc["A", "A"] == pytest.approx(1.0)


def test_correlation_matrix_identifies_anticorrelated_members():
    returns_a = [0.01, -0.02, 0.03, 0.0, -0.01]
    returns_b = [-r for r in returns_a]
    results = [_make_result("A", returns_a), _make_result("B", returns_b)]

    matrix = correlation_matrix(results)

    assert matrix.loc["A", "B"] == pytest.approx(-1.0)
