import numpy as np
import pandas as pd
import pytest

from src.analysis.monte_carlo import (
    MonteCarloAnalyzer,
    _block_bootstrap_matrix,
    _perturb_prices,
    _simple_bootstrap_matrix,
)
from src.backtest.engine import Backtester
from src.backtest.result import BacktestResult
from src.strategies.base import Strategy


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


def _make_result(returns_values, initial_capital=1000.0, strategy_name="Test"):
    index = pd.date_range("2022-01-01", periods=len(returns_values), freq="D", name="timestamp")
    returns = pd.Series(returns_values, index=index, dtype=float)
    equity = initial_capital * (1 + returns).cumprod()
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


class _FixedSignalStrategy(Strategy):
    def __init__(self, signal):
        self._signal = signal

    @property
    def name(self) -> str:
        return "Fixed"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = self._signal
        return out


def test_block_bootstrap_with_block_length_one_matches_simple_bootstrap():
    n, n_trials = 30, 200
    simple = _simple_bootstrap_matrix(n, n_trials, np.random.default_rng(7))
    block = _block_bootstrap_matrix(n, 1, n_trials, np.random.default_rng(7))

    np.testing.assert_array_equal(simple, block)


def test_constant_daily_return_produces_identical_equity_across_all_trials():
    result = _make_result([0.01] * 40)
    analyzer = MonteCarloAnalyzer(n_trials=200, seed=1, block_length=5)

    mc = analyzer.bootstrap(result)

    assert np.allclose(mc.final_equity, mc.final_equity[0])
    assert np.allclose(mc.max_drawdown, mc.max_drawdown[0])
    # constant positive returns never draw down and never finish below the initial capital
    assert mc.max_drawdown[0] == pytest.approx(0.0)
    assert mc.prob_below_initial_capital() == pytest.approx(0.0)


def test_constant_negative_return_always_finishes_below_initial_capital():
    result = _make_result([-0.01] * 40)
    analyzer = MonteCarloAnalyzer(n_trials=200, seed=1, block_length=5)

    mc = analyzer.bootstrap(result)

    assert mc.prob_below_initial_capital() == pytest.approx(1.0)


def test_percentiles_are_monotonically_ordered():
    rng = np.random.default_rng(3)
    returns = rng.normal(loc=0.0005, scale=0.02, size=200)
    result = _make_result(returns)
    analyzer = MonteCarloAnalyzer(n_trials=500, seed=9, block_length=10)

    mc = analyzer.bootstrap(result)

    equity_pcts = mc.final_equity_percentiles()
    drawdown_pcts = mc.max_drawdown_percentiles()

    assert equity_pcts[5] <= equity_pcts[25] <= equity_pcts[50] <= equity_pcts[75] <= equity_pcts[95]
    assert drawdown_pcts[5] <= drawdown_pcts[25] <= drawdown_pcts[50] <= drawdown_pcts[75] <= drawdown_pcts[95]


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(3)
    returns = rng.normal(loc=0.0, scale=0.01, size=100)
    result = _make_result(returns)

    mc_a = MonteCarloAnalyzer(n_trials=300, seed=11, block_length=15).bootstrap(result)
    mc_b = MonteCarloAnalyzer(n_trials=300, seed=11, block_length=15).bootstrap(result)

    np.testing.assert_array_equal(mc_a.final_equity, mc_b.final_equity)
    np.testing.assert_array_equal(mc_a.max_drawdown, mc_b.max_drawdown)


def test_noise_perturbation_with_zero_std_reproduces_actual_result_every_trial():
    close = [100, 102, 101, 105, 108, 107, 110]
    df = _make_ohlcv(close)
    strategy = _FixedSignalStrategy([0, 1, 1, 0, 1, 1, 0])
    backtester = Backtester(fee=0.0, initial_capital=1000.0, slippage_bps=0.0)
    analyzer = MonteCarloAnalyzer(n_trials=20, seed=5, noise_std=0.0)

    result = analyzer.noise_robustness(df, strategy, backtester)

    assert np.allclose(result.final_equity, result.actual_final_equity)


def test_noise_perturbation_preserves_ohlc_length_and_columns():
    close = [100, 101, 99, 103]
    df = _make_ohlcv(close)
    perturbed = _perturb_prices(df, noise_std=0.01, rng=np.random.default_rng(0))

    assert list(perturbed.columns) == list(df.columns)
    assert len(perturbed) == len(df)
    # volume must be left untouched by the price perturbation
    pd.testing.assert_series_equal(perturbed["volume"], df["volume"])
