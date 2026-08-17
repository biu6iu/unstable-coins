from __future__ import annotations
import numpy as np
import pandas as pd

from src.backtest.result import BacktestResult
from src.evaluation.metrics import ANNUALISATION_FACTOR


def simple_bootstrap_matrix(n: int, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """iid resampling"""
    return rng.integers(0, n, size=(n_trials, n))


def block_bootstrap_matrix(n: int, block_length: int, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """resamples contiguous blocks so volatility clustering survives the resampling"""
    if block_length <= 1:
        return simple_bootstrap_matrix(n, n_trials, rng)

    # ceil(n / block_length)
    n_blocks = -(-n // block_length)
    block_starts = rng.integers(0, n - block_length + 1, size=(n_trials, n_blocks))
    offsets = np.arange(block_length)
    indices = block_starts[:, :, None] + offsets[None, None, :]
    return indices.reshape(n_trials, n_blocks * block_length)[:, :n]


def trial_sharpes(sampled_returns: np.ndarray) -> np.ndarray:
    """annualised Sharpe of every bootstrap trial"""
    # ddof=1 to match pandas std
    trial_std = sampled_returns.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(trial_std > 0, sampled_returns.mean(axis=1) / trial_std * np.sqrt(ANNUALISATION_FACTOR), np.nan)


def paired_returns(result_a: BacktestResult, result_b: BacktestResult) -> tuple[np.ndarray, np.ndarray]:
    """restricts two strategies to the bars they share, since comparing them over different periods proves nothing"""
    aligned = pd.DataFrame({"a": result_a.strategy_returns, "b": result_b.strategy_returns}).dropna()

    if aligned.empty:
        raise ValueError(
            f"{result_a.strategy_name} and {result_b.strategy_name} share no overlapping bars to compare on"
        )
    return aligned["a"].to_numpy(), aligned["b"].to_numpy()
