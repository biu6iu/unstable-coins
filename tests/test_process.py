import numpy as np
import pytest

from src.stats.estimate import HypothesisTest
from src.stats.process import adf_test, kpss_test


def _stationary_ar1(n: int, phi: float = 0.3, seed: int = 1) -> np.ndarray:
    """a mean-reverting AR(1) series, the known-stationary case both tests should agree on"""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(scale=0.01, size=n)
    values = np.empty(n)
    values[0] = shocks[0]
    for t in range(1, n):
        values[t] = phi * values[t - 1] + shocks[t]
    return values


def _random_walk(n: int, seed: int = 2) -> np.ndarray:
    """cumulative sum of iid shocks, the known-unit-root case both tests should agree on"""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(scale=0.01, size=n))


def test_adf_rejects_the_unit_root_on_a_stationary_series():
    result = adf_test(_stationary_ar1(1000))
    assert isinstance(result, HypothesisTest)
    assert result.p_value < 0.05


def test_adf_cannot_reject_the_unit_root_on_a_random_walk():
    result = adf_test(_random_walk(1000))
    assert result.p_value > 0.05


def test_kpss_cannot_reject_stationarity_on_a_stationary_series():
    result = kpss_test(_stationary_ar1(1000))
    assert result.p_value > 0.05


def test_kpss_rejects_stationarity_on_a_random_walk():
    result = kpss_test(_random_walk(1000))
    assert result.p_value < 0.05


def test_adf_and_kpss_state_complementary_nulls():
    adf, kpss_result = adf_test(_stationary_ar1(500)), kpss_test(_stationary_ar1(500))
    assert "unit root" in adf.null
    assert "stationary" in kpss_result.null
    assert adf.null != kpss_result.null


def test_too_few_observations_is_rejected():
    with pytest.raises(ValueError):
        adf_test(np.array([0.01, 0.02, -0.01]))
    with pytest.raises(ValueError):
        kpss_test(np.array([0.01, 0.02, -0.01]))
