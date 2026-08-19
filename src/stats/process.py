from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from src.stats.estimate import HypothesisTest


def _clean(returns: pd.Series | np.ndarray) -> np.ndarray:
    """the engine leaves the first bar without a return, and both tests below need a plain array"""
    values = pd.Series(returns).dropna().to_numpy(dtype=float)
    if len(values) < 8:
        raise ValueError(f"need at least 8 observations for a stationarity test, got {len(values)}")
    return values


def adf_test(returns: pd.Series | np.ndarray, regression: str = "c") -> HypothesisTest:
    """Augmented Dickey-Fuller: null is a unit root, so rejecting it is evidence the series is stationary"""
    values = _clean(returns)
    statistic, p_value, *_ = adfuller(values, regression=regression)
    verdict = "rejects the unit root (consistent with stationarity)" if p_value < 0.05 else "cannot reject the unit root"
    return HypothesisTest(
        statistic=float(statistic),
        p_value=float(p_value),
        null="series has a unit root (is non-stationary)",
        conclusion=f"ADF {verdict}, p = {p_value:.4f}",
    )


def kpss_test(returns: pd.Series | np.ndarray, regression: str = "c") -> HypothesisTest:
    """KPSS: null is stationarity, the complement of ADF's null, so together they bound the answer instead of relying on one test's assumptions"""
    values = _clean(returns)
    with warnings.catch_warnings():
        # statsmodels warns when the true p-value falls outside its lookup table's range and clips it instead
        warnings.simplefilter("ignore", InterpolationWarning)
        statistic, p_value, *_ = kpss(values, regression=regression, nlags="auto")
    verdict = "rejects stationarity" if p_value < 0.05 else "cannot reject stationarity"
    return HypothesisTest(
        statistic=float(statistic),
        p_value=float(p_value),
        null="series is stationary",
        conclusion=f"KPSS {verdict}, p = {p_value:.4f}",
    )
