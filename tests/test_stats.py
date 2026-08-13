"""Tests for the statistical inference layer."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import ANNUALISATION_FACTOR, sharpe_ratio
from src.stats.estimate import Estimate, HypothesisTest
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
