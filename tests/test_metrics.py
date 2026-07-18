"""Tests for PerformanceMetrics - checked against small hand-computed examples."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.result import BacktestResult
from src.evaluation.metrics import ANNUALISATION_FACTOR, PerformanceMetrics


def _make_result(equity_values, returns_values=None, trade_count=1, strategy_name="Test"):
    index = pd.date_range("2022-01-01", periods=len(equity_values), freq="D", name="timestamp")
    equity = pd.Series(equity_values, index=index, dtype=float)
    returns = (
        equity.pct_change().fillna(0)
        if returns_values is None
        else pd.Series(returns_values, index=index, dtype=float)
    )
    df = pd.DataFrame({"close": equity_values}, index=index)
    positions = pd.Series(0, index=index)
    return BacktestResult(
        df=df,
        positions=positions,
        strategy_returns=returns,
        equity_curve=equity,
        trade_count=trade_count,
        strategy_name=strategy_name,
    )


def test_total_return_matches_hand_computed_value():
    result = _make_result([100, 110, 121])  # +10% then +10% -> +21% total
    metrics = PerformanceMetrics().compute(result)

    assert metrics["total_return"] == pytest.approx(0.21)


def test_max_drawdown_matches_hand_computed_value():
    # Peak at 120, trough at 90 -> drawdown of 90/120 - 1 = -25%.
    result = _make_result([100, 120, 90, 100])
    metrics = PerformanceMetrics().compute(result)

    assert metrics["max_drawdown"] == pytest.approx(90 / 120 - 1)


def test_annualised_volatility_matches_hand_computed_value():
    returns_values = [0.0, 0.01, -0.02, 0.015, -0.005]
    result = _make_result([100] * len(returns_values), returns_values=returns_values)
    metrics = PerformanceMetrics().compute(result)

    expected = pd.Series(returns_values).std() * np.sqrt(ANNUALISATION_FACTOR)
    assert metrics["annualised_volatility"] == pytest.approx(expected)


def test_sharpe_matches_hand_computed_value():
    returns_values = [0.0, 0.02, -0.01, 0.03, -0.02]
    result = _make_result([100] * len(returns_values), returns_values=returns_values)
    metrics = PerformanceMetrics().compute(result)

    returns = pd.Series(returns_values)
    expected = (returns.mean() / returns.std()) * np.sqrt(ANNUALISATION_FACTOR)
    assert metrics["sharpe"] == pytest.approx(expected)


def test_sharpe_is_nan_when_volatility_is_zero():
    returns_values = [0.01] * 5  # constant returns -> zero std
    result = _make_result([100] * 5, returns_values=returns_values)
    metrics = PerformanceMetrics().compute(result)

    assert np.isnan(metrics["sharpe"])


def test_cagr_matches_hand_computed_value_when_years_equals_one():
    n = ANNUALISATION_FACTOR  # exactly one year of daily bars -> years = 1
    equity = np.linspace(100, 121, n)
    result = _make_result(list(equity))
    metrics = PerformanceMetrics().compute(result)

    # With years = 1, CAGR reduces to the simple total return.
    assert metrics["cagr"] == pytest.approx(121 / 100 - 1)


def test_compute_carries_strategy_name_and_trade_count():
    result = _make_result([100, 110], trade_count=3, strategy_name="MyStrategy")
    metrics = PerformanceMetrics().compute(result)

    assert metrics["strategy"] == "MyStrategy"
    assert metrics["trade_count"] == 3


def test_compare_prints_a_row_per_strategy(capsys):
    results = [
        _make_result([100, 110], strategy_name="A"),
        _make_result([100, 90], strategy_name="B"),
    ]

    table = PerformanceMetrics().compare(results)
    captured = capsys.readouterr()

    assert "A" in table and "B" in table
    assert table in captured.out
