from __future__ import annotations
import math
from scipy import stats

from src.evaluation.metrics import ANNUALISATION_FACTOR


def _delta_sharpe_variance(corr: float, sr: float, years: float) -> float:
    """var(delta_SR) ~= 2*(1-rho)*(1+SR^2/2)/years, the annualised Sharpe difference's sampling variance"""
    if corr >= 1:
        raise ValueError(f"correlation must be below 1 for a nonzero difference variance, got {corr}")
    if years <= 0:
        raise ValueError(f"years must be positive, got {years}")
    return 2 * (1 - corr) * (1 + sr**2 / 2) / years


def sharpe_diff_power(delta: float, corr: float, sr: float, n_periods: int, alpha: float = 0.05) -> float:
    """probability that a true Sharpe difference of `delta` is detected with `n_periods` bars of daily data"""
    years = n_periods / ANNUALISATION_FACTOR
    se = math.sqrt(_delta_sharpe_variance(corr, sr, years))
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf(abs(delta) / se - z_alpha))


def required_periods(delta: float, corr: float, sr: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """bars of daily data needed to detect `delta` at the requested power, inverting sharpe_diff_power"""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    se_required = abs(delta) / (z_alpha + z_power)
    years = _delta_sharpe_variance_years(corr, sr, se_required**2)
    return math.ceil(years * ANNUALISATION_FACTOR)


def detectable_effect(n_periods: int, corr: float, sr: float, power: float = 0.8, alpha: float = 0.05) -> float:
    """the smallest Sharpe difference this much data could resolve at the requested power"""
    years = n_periods / ANNUALISATION_FACTOR
    se = math.sqrt(_delta_sharpe_variance(corr, sr, years))
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    return se * (z_alpha + z_power)


def _delta_sharpe_variance_years(corr: float, sr: float, variance: float) -> float:
    """years of data whose difference variance equals `variance`, the inverse of _delta_sharpe_variance"""
    if corr >= 1:
        raise ValueError(f"correlation must be below 1 for a finite sample size, got {corr}")
    return 2 * (1 - corr) * (1 + sr**2 / 2) / variance
