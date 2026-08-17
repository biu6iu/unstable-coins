"""Tests for the statistical inference layer."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.backtest.result import BacktestResult
from src.evaluation.metrics import ANNUALISATION_FACTOR, sharpe_ratio
from src.stats.attribution import alpha_beta
from src.stats.estimate import Estimate, HypothesisTest
from src.stats.multiple_testing import deflated_sharpe_ratio, expected_max_sharpe, reality_check
from src.stats.sharpe import (
    effective_sample_size,
    lo_annualisation_factor,
    newey_west_lags,
    sharpe_estimate,
    sharpe_se_hac,
    sharpe_se_iid,
    sharpe_se_nonnormal,
)

ROOT_Q = np.sqrt(ANNUALISATION_FACTOR)


def test_from_se_builds_a_symmetric_interval_around_the_estimate():
    estimate = Estimate.from_se(value=1.0, se=0.5, method="iid")

    assert estimate.value - estimate.ci_low == pytest.approx(estimate.ci_high - estimate.value)
    # the textbook 95% width, so a wrong z would be caught rather than merely look plausible
    assert estimate.ci_low == pytest.approx(1.0 - 1.959964 * 0.5, abs=1e-5)
    assert estimate.ci_high == pytest.approx(1.0 + 1.959964 * 0.5, abs=1e-5)


def test_higher_confidence_widens_the_interval():
    narrow = Estimate.from_se(value=1.0, se=0.5, method="iid", confidence=0.90)
    wide = Estimate.from_se(value=1.0, se=0.5, method="iid", confidence=0.99)

    assert wide.ci_low < narrow.ci_low
    assert wide.ci_high > narrow.ci_high


def test_excludes_reports_whether_the_interval_rules_out_a_null():
    clear = Estimate.from_se(value=1.0, se=0.1, method="iid")
    noisy = Estimate.from_se(value=1.0, se=5.0, method="iid")

    assert clear.excludes(0.0)
    assert not noisy.excludes(0.0)
    # an estimate can be far from zero yet still consistent with a different null
    assert not clear.excludes(1.05)


def test_zero_standard_error_leaves_a_degenerate_interval():
    estimate = Estimate.from_se(value=2.0, se=0.0, method="exact")

    assert estimate.ci_low == estimate.ci_high == 2.0
    assert estimate.excludes(0.0)


def test_hypothesis_test_significance_respects_alpha():
    result = HypothesisTest(statistic=2.1, p_value=0.03, null="sharpe = 0", conclusion="reject")

    assert result.is_significant()
    assert not result.is_significant(alpha=0.01)


def _normal_returns(n=2000, sharpe=0.05, seed=0):
    """iid normal returns with a known per-period Sharpe"""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(sharpe * 0.02, 0.02, n))


def _ar1_returns(phi, n=4000, seed=1):
    """returns with injected serial correlation, the case eta(q) and the HAC SE exist for"""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0, 0.02, n)
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = phi * values[i - 1] + shocks[i]
    return pd.Series(values + 0.001)


def test_iid_se_matches_the_empirical_spread_of_repeated_samples():
    n, sharpe = 1000, 0.05
    rng = np.random.default_rng(11)
    samples = rng.normal(sharpe * 0.02, 0.02, size=(4000, n))
    empirical = (samples.mean(axis=1) / samples.std(axis=1, ddof=1)).std()

    assert sharpe_se_iid(sharpe, n) == pytest.approx(empirical, rel=0.05)


def test_annualised_se_matches_the_empirical_spread_so_it_is_scaled_only_once():
    n = 3280
    rng = np.random.default_rng(3)
    samples = rng.normal(0.788 / ROOT_Q, 1.0, size=(3000, n))
    empirical = (samples.mean(axis=1) / samples.std(axis=1, ddof=1) * ROOT_Q).std()

    assert sharpe_estimate(samples[0], "iid").se == pytest.approx(empirical, rel=0.05)


def test_nonnormal_se_converges_on_the_iid_se_for_normal_returns():
    returns = _normal_returns(n=5000, seed=4)
    per_period_sharpe = sharpe_ratio(returns) / ROOT_Q

    assert sharpe_se_nonnormal(returns) == pytest.approx(sharpe_se_iid(per_period_sharpe, len(returns)), rel=0.02)


def test_nonnormal_se_exceeds_the_iid_se_when_returns_are_skewed_and_fat_tailed():
    rng = np.random.default_rng(5)
    # a t(4) draw with a negative shift is fat-tailed and left-skewed, the shape the correction is for
    returns = pd.Series(0.05 + rng.standard_t(4, 3000) - 0.4 * rng.standard_t(4, 3000) ** 2)
    per_period_sharpe = sharpe_ratio(returns) / ROOT_Q

    assert sharpe_se_nonnormal(returns) > sharpe_se_iid(per_period_sharpe, len(returns))


def test_hac_se_exceeds_the_iid_se_under_positive_autocorrelation():
    returns = _ar1_returns(0.3)
    per_period_sharpe = sharpe_ratio(returns) / ROOT_Q

    assert sharpe_se_hac(returns) > 1.2 * sharpe_se_iid(per_period_sharpe, len(returns))


def test_hac_and_iid_se_agree_on_white_noise():
    returns = _ar1_returns(0.0)
    per_period_sharpe = sharpe_ratio(returns) / ROOT_Q

    assert sharpe_se_hac(returns) == pytest.approx(sharpe_se_iid(per_period_sharpe, len(returns)), rel=0.05)


def test_lo_factor_falls_below_root_q_under_positive_autocorrelation():
    assert lo_annualisation_factor(_ar1_returns(0.3)) < 0.85 * ROOT_Q
    assert lo_annualisation_factor(_ar1_returns(-0.3)) > 1.15 * ROOT_Q
    assert lo_annualisation_factor(_ar1_returns(0.0)) == pytest.approx(ROOT_Q, rel=0.05)


def test_hac_sharpe_is_smaller_than_the_naive_sharpe_under_positive_autocorrelation():
    returns = _ar1_returns(0.3)

    assert sharpe_estimate(returns, "hac").value < sharpe_estimate(returns, "iid").value


def test_effective_sample_size_shrinks_under_positive_autocorrelation():
    n = 4000
    assert effective_sample_size(_ar1_returns(0.3, n=n)) < 0.7 * n
    assert effective_sample_size(_ar1_returns(0.0, n=n)) == pytest.approx(n, rel=0.1)


def test_point_estimate_reuses_metrics_sharpe_ratio():
    returns = _normal_returns(seed=6)

    for method in ("iid", "nonnormal"):
        assert sharpe_estimate(returns, method).value == pytest.approx(sharpe_ratio(returns))


def test_sharpe_estimate_hands_its_se_to_the_interval_unaltered():
    estimate = sharpe_estimate(_normal_returns(seed=7), "nonnormal")

    assert estimate.ci_high - estimate.ci_low == pytest.approx(2 * 1.959964 * estimate.se, rel=1e-4)


def test_leading_nan_is_dropped_rather_than_poisoning_every_moment():
    returns = _normal_returns(seed=8)
    with_nan = pd.concat([pd.Series([np.nan]), returns], ignore_index=True)

    assert sharpe_estimate(with_nan, "nonnormal").value == pytest.approx(sharpe_estimate(returns, "nonnormal").value)


def test_newey_west_lags_grows_with_the_sample():
    assert newey_west_lags(100) < newey_west_lags(10000)
    assert newey_west_lags(3280) == 8


def test_unknown_method_and_degenerate_input_are_rejected():
    with pytest.raises(ValueError, match="unknown Sharpe SE method"):
        sharpe_estimate(_normal_returns(), "bootstrap")

    with pytest.raises(ValueError, match="at least 3 returns"):
        sharpe_estimate(pd.Series([0.01, np.nan]))


def test_zero_variance_returns_give_no_sharpe_rather_than_a_divide_by_zero():
    flat = pd.Series(np.zeros(100))

    assert np.isnan(sharpe_estimate(flat, "nonnormal").value)
    assert np.isnan(sharpe_se_hac(flat))


def test_expected_max_sharpe_grows_with_the_width_of_the_search():
    widths = [expected_max_sharpe(n, 0.5) for n in (2, 10, 100, 1000)]

    assert widths == sorted(widths)
    assert all(width > 0 for width in widths)


def test_expected_max_sharpe_scales_with_the_dispersion_of_the_trials():
    assert expected_max_sharpe(100, 1.0) == pytest.approx(2 * expected_max_sharpe(100, 0.5))
    # a search over identical candidates cannot get lucky, however wide it is
    assert expected_max_sharpe(1000, 0.0) == 0.0


def test_a_single_trial_has_no_maximum_to_inflate():
    assert expected_max_sharpe(1, 0.5) == 0.0


def test_expected_max_sharpe_rejects_an_impossible_search():
    with pytest.raises(ValueError, match="at least one trial"):
        expected_max_sharpe(0, 0.5)

    with pytest.raises(ValueError, match="cannot be negative"):
        expected_max_sharpe(10, -0.1)


def test_deflated_sharpe_reduces_to_the_ordinary_sharpe_p_value_for_one_trial():
    returns = _normal_returns(n=3000, sharpe=0.05, seed=20)
    estimate = sharpe_estimate(returns, "nonnormal")
    ordinary_p = float(stats.norm.sf(estimate.value / estimate.se))

    result = deflated_sharpe_ratio(returns, n_trials=1, trial_sr_std=0.5)

    assert result.p_value == pytest.approx(ordinary_p)


def test_deflation_makes_a_sharpe_harder_to_believe_as_the_search_widens():
    returns = _normal_returns(n=3000, sharpe=0.05, seed=21)
    p_values = [deflated_sharpe_ratio(returns, n, 0.4).p_value for n in (1, 10, 100, 1000)]

    assert p_values == sorted(p_values)


def test_a_lucky_sharpe_from_a_wide_search_does_not_survive_deflation():
    # a genuine Sharpe of ~1.8 annualised, but found by a 500-wide search whose trials scatter by 0.6
    returns = _normal_returns(n=3000, sharpe=1.8 / ROOT_Q, seed=22)

    assert deflated_sharpe_ratio(returns, n_trials=1, trial_sr_std=0.6).is_significant()
    assert not deflated_sharpe_ratio(returns, n_trials=500, trial_sr_std=0.6).is_significant()


def test_deflated_sharpe_reports_the_benchmark_it_tested_against():
    result = deflated_sharpe_ratio(_normal_returns(seed=23), n_trials=100, trial_sr_std=0.5)

    assert f"{expected_max_sharpe(100, 0.5):.3f}" in result.null


def test_deflating_a_flat_return_series_is_undefined_rather_than_significant():
    result = deflated_sharpe_ratio(pd.Series(np.zeros(100)), n_trials=10, trial_sr_std=0.5)

    assert np.isnan(result.p_value)
    assert not result.is_significant()


def _candidate_field(n_with_edge=0, edge=0.001, n_candidates=20, n_bars=1500, seed=0):
    """a benchmark plus candidates that track it with independent noise, some given a genuine constant edge"""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n_bars, freq="D", name="timestamp")
    benchmark = pd.Series(rng.normal(0.0005, 0.02, n_bars), index=index)

    candidates = {}
    for i in range(n_candidates):
        tracked = benchmark.to_numpy() + rng.normal(0, 0.01, n_bars)
        candidates[f"s{i}"] = pd.Series(tracked + (edge if i < n_with_edge else 0.0), index=index)
    return pd.DataFrame(candidates), benchmark


def test_a_field_of_pure_noise_strategies_does_not_beat_the_benchmark():
    candidates, benchmark = _candidate_field(seed=1)

    assert not reality_check(candidates, benchmark, n_trials=2000).is_significant()


def test_a_genuine_constant_edge_is_found():
    candidates, benchmark = _candidate_field(n_with_edge=1, seed=1)
    result = reality_check(candidates, benchmark, n_trials=2000)

    assert result.is_significant()
    assert "s0" in result.conclusion


def test_reality_check_rejects_a_winner_the_naive_per_candidate_test_would_believe():
    # seed 2 is a search over pure noise whose luckiest candidate looks significant on its own
    candidates, benchmark = _candidate_field(seed=2)
    excess = candidates.sub(benchmark, axis=0)
    best_naive_p = min(float(stats.ttest_1samp(excess[c], 0, alternative="greater").pvalue) for c in excess)

    assert best_naive_p < 0.01
    assert not reality_check(candidates, benchmark, n_trials=2000).is_significant()


def test_duplicating_a_candidate_cannot_change_the_verdict():
    """every candidate must see the same resampled bars, or a duplicate would draw its own and shift the maximum"""
    candidates, benchmark = _candidate_field(n_candidates=5, seed=3)
    padded = candidates.assign(s0_again=candidates["s0"])

    assert reality_check(padded, benchmark, n_trials=1000).p_value == pytest.approx(
        reality_check(candidates, benchmark, n_trials=1000).p_value
    )


def test_reality_check_is_reproducible_and_seed_dependent():
    candidates, benchmark = _candidate_field(n_candidates=5, seed=4)
    p_value = reality_check(candidates, benchmark, n_trials=500, seed=7).p_value

    assert reality_check(candidates, benchmark, n_trials=500, seed=7).p_value == p_value
    assert reality_check(candidates, benchmark, n_trials=500, seed=8).p_value != p_value


def test_a_single_candidate_is_a_valid_if_uncorrected_search():
    candidates, benchmark = _candidate_field(n_with_edge=1, n_candidates=1, seed=5)

    assert reality_check(candidates, benchmark, n_trials=1000).is_significant()


def test_reality_check_rejects_an_empty_field_and_a_disjoint_benchmark():
    candidates, benchmark = _candidate_field(n_candidates=3, seed=6)

    with pytest.raises(ValueError, match="at least one candidate"):
        reality_check(pd.DataFrame(index=candidates.index), benchmark)

    elsewhere = benchmark.copy()
    elsewhere.index = benchmark.index + pd.Timedelta(days=10_000)
    with pytest.raises(ValueError, match="no overlapping bars"):
        reality_check(candidates, elsewhere)


def _make_result(returns_values, strategy_name="Test"):
    """a BacktestResult whose non-returns fields are never touched by paired_returns/alpha_beta"""
    index = pd.date_range("2020-01-01", periods=len(returns_values), freq="D", name="timestamp")
    returns = pd.Series(returns_values, index=index, dtype=float)
    zeros = pd.Series(0.0, index=index)
    equity = 1000.0 * (1 + returns).cumprod()
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


def test_alpha_beta_recovers_a_known_linear_relationship():
    rng = np.random.default_rng(30)
    n = 3000
    benchmark_returns = rng.normal(0.0005, 0.02, n)
    strategy_returns = 2 * benchmark_returns + 0.001

    result = alpha_beta(_make_result(strategy_returns, "Strategy"), _make_result(benchmark_returns, "Benchmark"))

    assert result["beta"].value == pytest.approx(2.0, abs=0.05)
    assert result["alpha"].value == pytest.approx(0.001 * ANNUALISATION_FACTOR, rel=0.05)
    assert result["alpha"].excludes(0.0)
    assert result["beta"].excludes(0.0)


def test_alpha_beta_finds_no_alpha_when_the_strategy_is_pure_beta():
    rng = np.random.default_rng(31)
    benchmark_returns = rng.normal(0.0005, 0.02, 3000)

    result = alpha_beta(_make_result(1.5 * benchmark_returns, "Strategy"), _make_result(benchmark_returns, "Benchmark"))

    assert not result["alpha"].excludes(0.0)


def test_alpha_beta_uses_hac_standard_errors_that_widen_under_autocorrelation():
    strategy_returns = _ar1_returns(0.3, n=3000, seed=32).to_numpy()
    benchmark_returns = _ar1_returns(0.3, n=3000, seed=33).to_numpy()

    hac = alpha_beta(_make_result(strategy_returns), _make_result(benchmark_returns), max_lags=8)
    iid = alpha_beta(_make_result(strategy_returns), _make_result(benchmark_returns), max_lags=0)

    assert hac["alpha"].se > iid["alpha"].se


def test_alpha_beta_restricts_to_shared_bars():
    rng = np.random.default_rng(34)
    benchmark_returns = rng.normal(0.0005, 0.02, 500)
    strategy = _make_result(2 * benchmark_returns + 0.001, "Strategy")
    benchmark = _make_result(benchmark_returns, "Benchmark")
    benchmark.strategy_returns.index = benchmark.strategy_returns.index + pd.Timedelta(days=10_000)

    with pytest.raises(ValueError, match="no overlapping bars"):
        alpha_beta(strategy, benchmark)
