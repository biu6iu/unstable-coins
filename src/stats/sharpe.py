from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.metrics import ANNUALISATION_FACTOR, sharpe_ratio
from src.stats.estimate import Estimate

SE_METHODS = ("iid", "nonnormal", "hac")


def _clean(returns: pd.Series | np.ndarray) -> np.ndarray:
    """the engine leaves the first bar without a return, and every formula here divides by n"""
    values = pd.Series(returns).dropna().to_numpy(dtype=float)
    if len(values) < 3:
        raise ValueError(f"need at least 3 returns to estimate a Sharpe standard error, got {len(values)}")
    return values


def _per_period_sharpe(values: np.ndarray) -> float:
    """un-annualises metrics.sharpe_ratio so there is still only one owner of the point estimate"""
    return sharpe_ratio(pd.Series(values)) / np.sqrt(ANNUALISATION_FACTOR)


def newey_west_lags(n: int) -> int:
    """the standard automatic truncation floor(4*(T/100)^(2/9))"""
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def _autocorrelations(values: np.ndarray, max_lags: int | None) -> np.ndarray:
    """rho_1 .. rho_L, truncated because sample autocorrelations at long lags are mostly noise"""
    n = len(values)
    lags = newey_west_lags(n) if max_lags is None else max_lags
    lags = min(lags, n - 1)

    centred = values - values.mean()
    denominator = float(centred @ centred)
    if denominator == 0:
        return np.zeros(lags)
    return np.array([float(centred[k:] @ centred[:-k]) / denominator for k in range(1, lags + 1)])


def _variance_inflation(values: np.ndarray, max_lags: int | None) -> float:
    """1 + 2*sum(rho_k), the factor by which serial correlation inflates the variance of the sample mean"""
    return 1 + 2 * float(_autocorrelations(values, max_lags).sum())


def effective_sample_size(returns: pd.Series | np.ndarray, max_lags: int | None = None) -> float:
    """the number of independent observations n autocorrelated bars are actually worth"""
    values = _clean(returns)
    inflation = _variance_inflation(values, max_lags)
    return len(values) / inflation if inflation > 0 else np.nan


def sharpe_se_iid(sr: float, n: int) -> float:
    """normal-theory baseline sqrt((1 + sr^2/2)/n), in the same per-period units as sr"""
    return float(np.sqrt((1 + sr**2 / 2) / n))


def sharpe_se_nonnormal(returns: pd.Series | np.ndarray) -> float:
    """Mertens' skew and kurtosis correction, the one that matters at this market's excess kurtosis of 8.7"""
    values = _clean(returns)
    sr = _per_period_sharpe(values)
    if np.isnan(sr):
        return np.nan

    skew = float(stats.skew(values, bias=False))
    # non-excess kurtosis, so that a normal sample's (kurt-1)/4 = 0.5 reproduces sharpe_se_iid exactly
    kurtosis = float(stats.kurtosis(values, fisher=False, bias=False))

    variance = (1 - skew * sr + (kurtosis - 1) / 4 * sr**2) / len(values)
    return float(np.sqrt(variance)) if variance > 0 else np.nan


def sharpe_se_hac(returns: pd.Series | np.ndarray, max_lags: int | None = None) -> float:
    """iid formula with the mean term widened by serial correlation, since autocorrelated bars carry less information"""
    values = _clean(returns)
    sr = _per_period_sharpe(values)
    if np.isnan(sr):
        return np.nan

    inflation = _variance_inflation(values, max_lags)
    if inflation <= 0:
        return np.nan
    return float(np.sqrt((inflation + sr**2 / 2) / len(values)))


def lo_annualisation_factor(returns: pd.Series | np.ndarray, max_lags: int | None = None, periods: int = ANNUALISATION_FACTOR) -> float:
    """Lo (2002)'s eta(q), which replaces sqrt(q) so autocorrelation is not laundered into a larger Sharpe"""
    values = _clean(returns)
    rho = _autocorrelations(values, max_lags)
    lags = np.arange(1, len(rho) + 1)

    # the variance of a q-period sum, not q times the variance of one period
    long_run = periods + 2 * float(((periods - lags) * rho).sum())
    return periods / np.sqrt(long_run) if long_run > 0 else np.nan


def sharpe_estimate(returns: pd.Series | np.ndarray, method: str = "nonnormal", max_lags: int | None = None) -> Estimate:
    """annualised Sharpe with an interval"""
    if method not in SE_METHODS:
        raise ValueError(f"unknown Sharpe SE method {method!r}, expected one of {SE_METHODS}")

    values = _clean(returns)
    sr = _per_period_sharpe(values)

    if method == "iid":
        se, scale = sharpe_se_iid(sr, len(values)), np.sqrt(ANNUALISATION_FACTOR)
    elif method == "nonnormal":
        se, scale = sharpe_se_nonnormal(values), np.sqrt(ANNUALISATION_FACTOR)
    else:
        se, scale = sharpe_se_hac(values, max_lags), lo_annualisation_factor(values, max_lags)

    return Estimate.from_se(value=sr * scale, se=se * scale, method=method)
