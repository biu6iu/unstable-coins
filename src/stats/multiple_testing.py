from __future__ import annotations
from collections.abc import Mapping
import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.metrics import ANNUALISATION_FACTOR
from src.stats.estimate import HypothesisTest
from src.stats.resampling import block_bootstrap_matrix
from src.stats.sharpe import sharpe_estimate

_BENCHMARK = "__benchmark__"


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


def _align_excess(candidate_returns: pd.DataFrame | Mapping[str, pd.Series], benchmark_returns: pd.Series) -> tuple[list[str], np.ndarray]:
    frame = pd.DataFrame(candidate_returns).copy()
    if frame.empty or not len(frame.columns):
        raise ValueError("a reality check needs at least one candidate strategy")

    frame[_BENCHMARK] = pd.Series(benchmark_returns)
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("the candidates and the benchmark share no overlapping bars to compare on")

    benchmark = frame.pop(_BENCHMARK).to_numpy(dtype=float)
    return list(frame.columns), frame.to_numpy(dtype=float) - benchmark[:, None]


def reality_check(candidate_returns: pd.DataFrame | Mapping[str, pd.Series], benchmark_returns: pd.Series, n_trials: int = 5000, block_length: int = 20, seed: int | None = 42,) -> HypothesisTest:
    """White's Reality Check: does the best candidate beat the benchmark once the search for it is priced in?"""
    names, excess = _align_excess(candidate_returns, benchmark_returns)
    n_bars = len(excess)
    root_t = np.sqrt(n_bars)

    mean_excess = excess.mean(axis=0)
    observed = root_t * mean_excess
    best = int(np.argmax(observed))

    rng = np.random.default_rng(seed)

    # one set of indices shared by every candidate so a trial cannot favour one by drawing it different bars
    indices = block_bootstrap_matrix(n_bars, block_length, n_trials, rng)

    trial_means = np.empty((n_trials, len(names)))
    for k in range(len(names)):
        trial_means[:, k] = excess[:, k][indices].mean(axis=1)

    # recentred on the observed means
    trial_max = (root_t * (trial_means - mean_excess)).max(axis=1)
    p_value = float(np.mean(trial_max >= observed[best]))

    return HypothesisTest(
        statistic=float(observed[best]),
        p_value=p_value,
        null=f"none of the {len(names)} candidates beats the benchmark: max true mean excess return <= 0",
        conclusion=_reality_conclusion(names[best], mean_excess[best], p_value),
    )


def _reality_conclusion(best: str, mean_excess: float, p_value: float) -> str:
    annualised = mean_excess * ANNUALISATION_FACTOR
    verdict = "beats" if p_value < 0.05 else "does not beat"
    return f"best candidate {best} ({annualised:+.1%} annualised excess) {verdict} the benchmark, p = {p_value:.3f}"


def _null(benchmark: float, n_trials: int) -> str:
    if n_trials == 1:
        return "true annualised Sharpe <= 0.000, with no search to correct for at one trial"
    return f"true annualised Sharpe <= {benchmark:.3f}, the best {n_trials} noise trials would be expected to find"


def _conclusion(observed: float, benchmark: float, p_value: float) -> str:
    if p_value < 0.05:
        return f"Sharpe {observed:.3f} survives deflation against {benchmark:.3f} (p = {p_value:.3f})"
    return f"Sharpe {observed:.3f} is not distinguishable from a search of this width finding {benchmark:.3f}"
