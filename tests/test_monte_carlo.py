import numpy as np
import pandas as pd
import pytest

from src.analysis.monte_carlo import MonteCarloAnalyzer, _perturb_prices
from src.backtest.engine import Backtester
from src.backtest.result import BacktestResult
from src.evaluation.metrics import sharpe_ratio
from src.stats.resampling import block_bootstrap_matrix, simple_bootstrap_matrix
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
    simple = simple_bootstrap_matrix(n, n_trials, np.random.default_rng(7))
    block = block_bootstrap_matrix(n, 1, n_trials, np.random.default_rng(7))

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


def test_sample_paths_shape_and_dates_match_result():
    result = _make_result([0.001] * 50)
    analyzer = MonteCarloAnalyzer(n_trials=30, seed=1, block_length=5)

    mc = analyzer.bootstrap(result, n_sample_paths=10)

    assert mc.sample_paths.shape == (10, 50)
    pd.testing.assert_index_equal(mc.dates, result.equity_curve.index)


def test_sample_paths_capped_at_n_trials_when_n_sample_paths_is_larger():
    result = _make_result([0.001] * 20)
    analyzer = MonteCarloAnalyzer(n_trials=5, seed=1, block_length=5)

    mc = analyzer.bootstrap(result, n_sample_paths=100)

    assert mc.sample_paths.shape == (5, 20)


def test_sample_paths_identical_under_constant_returns():
    result = _make_result([0.01] * 40)
    analyzer = MonteCarloAnalyzer(n_trials=200, seed=1, block_length=5)

    mc = analyzer.bootstrap(result, n_sample_paths=50)

    for path in mc.sample_paths:
        assert np.allclose(path, mc.sample_paths[0])


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


def test_bootstrapped_sharpe_matches_metrics_sharpe_when_trials_reproduce_the_history():
    # block_length == len(returns) leaves exactly one legal block start, so every
    # trial is the original series. Each trial's Sharpe must then equal
    # metrics.sharpe_ratio's: a bootstrapped Sharpe computed with a different
    # ddof would not be comparable with the headline number it brackets.
    rng = np.random.default_rng(1)
    result = _make_result(rng.normal(0.001, 0.02, size=60))
    mc = MonteCarloAnalyzer(n_trials=25, block_length=60).bootstrap(result)

    expected = sharpe_ratio(result.strategy_returns)
    assert mc.actual_sharpe == pytest.approx(expected)
    assert mc.sharpe == pytest.approx(np.full(25, expected))


def test_sharpe_percentiles_are_monotonically_ordered_and_bracket_the_actual():
    rng = np.random.default_rng(0)
    result = _make_result(rng.normal(0.001, 0.02, size=250))
    mc = MonteCarloAnalyzer(n_trials=500, block_length=20).bootstrap(result)

    percentiles = mc.sharpe_percentiles()
    assert [percentiles[p] for p in (5, 25, 50, 75, 95)] == sorted(
        percentiles[p] for p in (5, 25, 50, 75, 95)
    )
    assert percentiles[5] < mc.actual_sharpe < percentiles[95]


def test_sharpe_percentiles_survive_trials_with_no_variation():
    # A strategy that never moved has no Sharpe (nan), and nan trials must not
    # blank the percentile summary for the trials that do have one.
    result = _make_result([0.0] * 30)
    mc = MonteCarloAnalyzer(n_trials=20, block_length=1).bootstrap(result)

    assert np.isnan(mc.actual_sharpe)
    assert np.isnan(mc.sharpe).all()
    with pytest.warns(RuntimeWarning, match="All-NaN"):
        assert np.isnan(mc.sharpe_percentiles()[50])


# --- paired strategy comparison ------------------------------------------


def test_identical_results_give_an_exactly_zero_delta_distribution():
    returns = np.random.default_rng(0).normal(0.001, 0.02, size=200)
    a = _make_result(returns, strategy_name="A")
    b = _make_result(returns, strategy_name="B")

    comparison = MonteCarloAnalyzer(n_trials=200, block_length=20).compare_strategies(a, b)

    # both series read through the same resampled bars, so every trial cancels exactly
    assert np.all(comparison.sharpe_delta == 0.0)
    assert comparison.prob_a_beats_b() == 0.0
    assert not comparison.is_distinguishable()


def test_strictly_dominating_strategy_beats_the_other_in_every_trial():
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 0.02, size=200)
    # a shifts every bar's return up by a constant: same volatility, higher mean,
    # so a's Sharpe exceeds b's on any resampling of the bars whatsoever
    a = _make_result(base + 0.01, strategy_name="Better")
    b = _make_result(base, strategy_name="Worse")

    comparison = MonteCarloAnalyzer(n_trials=200, block_length=20).compare_strategies(a, b)

    assert comparison.prob_a_beats_b() == 1.0
    assert comparison.actual_delta > 0
    assert comparison.is_distinguishable()
    assert comparison.delta_percentiles()[5] > 0


def test_comparison_is_reproducible_for_a_given_seed():
    rng = np.random.default_rng(2)
    a = _make_result(rng.normal(0.001, 0.02, size=150), strategy_name="A")
    b = _make_result(rng.normal(0.001, 0.02, size=150), strategy_name="B")

    first = MonteCarloAnalyzer(n_trials=100, seed=7).compare_strategies(a, b)
    second = MonteCarloAnalyzer(n_trials=100, seed=7).compare_strategies(a, b)
    different_seed = MonteCarloAnalyzer(n_trials=100, seed=8).compare_strategies(a, b)

    assert np.array_equal(first.sharpe_delta, second.sharpe_delta)
    assert not np.array_equal(first.sharpe_delta, different_seed.sharpe_delta)


def test_pairing_is_tighter_than_two_independent_bootstraps():
    # the entire justification for pairing: both strategies face the same drawn
    # market, so the shared "which bars came up" variance cancels out of the
    # difference instead of compounding across two independent draws
    rng = np.random.default_rng(3)
    market = rng.normal(0.001, 0.02, size=300)
    a = _make_result(market + rng.normal(0.0, 0.002, size=300), strategy_name="A")
    b = _make_result(market + rng.normal(0.0, 0.002, size=300), strategy_name="B")

    analyzer = MonteCarloAnalyzer(n_trials=1000, block_length=20)
    paired = analyzer.compare_strategies(a, b).sharpe_delta
    independent = analyzer.bootstrap(a).sharpe - MonteCarloAnalyzer(
        n_trials=1000, seed=99, block_length=20
    ).bootstrap(b).sharpe

    assert np.std(paired) < np.std(independent)


def test_comparison_uses_only_the_bars_both_strategies_share():
    rng = np.random.default_rng(4)
    a = _make_result(rng.normal(0.001, 0.02, size=200), strategy_name="Long")
    # a walk-forward result covers fewer bars than a plain backtest of the same history
    b = _make_result(rng.normal(0.001, 0.02, size=200), strategy_name="Short")
    b.strategy_returns.iloc[:120] = np.nan

    comparison = MonteCarloAnalyzer(n_trials=50).compare_strategies(a, b)

    assert comparison.n_bars == 80


def test_comparison_rejects_strategies_with_no_overlapping_bars():
    a = _make_result(np.full(50, 0.01), strategy_name="A")
    b = _make_result(np.full(50, 0.01), strategy_name="B")
    b.strategy_returns.index = pd.date_range("2030-01-01", periods=50, freq="D", name="timestamp")

    with pytest.raises(ValueError, match="no overlapping bars"):
        MonteCarloAnalyzer(n_trials=10).compare_strategies(a, b)
