from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from src.stats.estimate import HypothesisTest
from src.stats.sharpe import sharpe_estimate


def expected_max_sharpe(n_trials: int, trial_sr_std: float) -> float:
    """the Sharpe a pure-noise search of this width is expected to produce, Bailey and Lopez de Prado (2014)"""
    if n_trials < 1:
        raise ValueError(f"a search has at least one trial, got {n_trials}")
    if trial_sr_std < 0:
        raise ValueError(f"trial Sharpe dispersion cannot be negative, got {trial_sr_std}")

    # one draw has no maximum to inflate, so the expectation is just the null mean of zero
    if n_trials == 1:
        return 0.0

    upper = float(stats.norm.ppf(1 - 1 / n_trials))
    inner = float(stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    return float(trial_sr_std * ((1 - np.euler_gamma) * upper + np.euler_gamma * inner))


def deflated_sharpe_ratio(returns: pd.Series | np.ndarray, n_trials: int, trial_sr_std: float) -> HypothesisTest:
    """
    probability the observed Sharpe beats the best a same-sized noise search would find
    
    DSR = 1 - p_value
    """
    observed = sharpe_estimate(returns, "nonnormal")
    benchmark = expected_max_sharpe(n_trials, trial_sr_std)

    if np.isnan(observed.value) or np.isnan(observed.se) or observed.se == 0:
        return HypothesisTest(
            statistic=np.nan,
            p_value=np.nan,
            null=_null(benchmark, n_trials),
            conclusion="no Sharpe to deflate: the returns have no usable variance",
        )

    statistic = (observed.value - benchmark) / observed.se
    p_value = float(stats.norm.sf(statistic))
    return HypothesisTest(
        statistic=statistic,
        p_value=p_value,
        null=_null(benchmark, n_trials),
        conclusion=_conclusion(observed.value, benchmark, p_value),
    )


def _null(benchmark: float, n_trials: int) -> str:
    if n_trials == 1:
        return "true annualised Sharpe <= 0.000, with no search to correct for at one trial"
    return f"true annualised Sharpe <= {benchmark:.3f}, the best {n_trials} noise trials would be expected to find"


def _conclusion(observed: float, benchmark: float, p_value: float) -> str:
    if p_value < 0.05:
        return f"Sharpe {observed:.3f} survives deflation against {benchmark:.3f} (p = {p_value:.3f})"
    return f"Sharpe {observed:.3f} is not distinguishable from a search of this width finding {benchmark:.3f}"
