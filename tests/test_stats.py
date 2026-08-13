"""Tests for the statistical inference layer."""

import pytest

from src.stats.estimate import Estimate, HypothesisTest


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
