"""Monte Carlo robustness analysis for backtest results.

CORE INSIGHT: a historical backtest is ONE draw from a distribution of
outcomes the strategy's daily behaviour could plausibly have produced.
Maximum drawdown is especially order-dependent - the same set of daily
returns, replayed in a different sequence, can front-load its losses
into one deep drawdown or spread them thin - so the realised max
drawdown in a single backtest typically UNDERSTATES the range of
drawdowns that behaviour could produce. Resampling turns this hidden
uncertainty into an explicit distribution instead of one historical
number.

METHOD AND ITS LIMITATION: bootstrap resampling draws returns with
replacement, which treats every day as independent. That is a real
simplification: if a strategy's returns are autocorrelated - as
trend-following returns typically are, since a position is held across
many similar bars in a row - independent resampling destroys that
structure and can understate genuine tail risk. Block bootstrap
resamples contiguous blocks of `block_length` bars instead of single
bars, preserving within-block autocorrelation, and is the
default-recommended variant here for exactly that reason.
`block_length=1` collapses block bootstrap to plain iid bootstrap (this
equivalence is verified directly in tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.engine import Backtester
from src.backtest.result import BacktestResult
from src.strategies.base import Strategy

PERCENTILES = (5, 25, 50, 75, 95)


def _percentile_summary(values: np.ndarray) -> dict[int, float]:
    return {p: float(np.percentile(values, p)) for p in PERCENTILES}


@dataclass
class MonteCarloResult:
    """Distributions of final equity and max drawdown from resampled
    trials of one strategy's realised daily returns, plus the single
    historical (actual) values for comparison."""

    final_equity: np.ndarray
    max_drawdown: np.ndarray
    actual_final_equity: float
    actual_max_drawdown: float
    initial_capital: float
    block_length: int

    def final_equity_percentiles(self) -> dict[int, float]:
        return _percentile_summary(self.final_equity)

    def max_drawdown_percentiles(self) -> dict[int, float]:
        return _percentile_summary(self.max_drawdown)

    def prob_below_initial_capital(self) -> float:
        return float(np.mean(self.final_equity < self.initial_capital))


@dataclass
class NoiseRobustnessResult:
    """Spread of final-equity outcomes when the same strategy is re-run on
    many small Gaussian perturbations of the input price series. A
    fragile, overfit strategy's performance degrades sharply under tiny
    perturbations; a robust one is largely unaffected."""

    final_equity: np.ndarray
    actual_final_equity: float
    noise_std: float

    def final_equity_percentiles(self) -> dict[int, float]:
        return _percentile_summary(self.final_equity)


def _simple_bootstrap_matrix(n: int, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """iid resampling: draw n indices with replacement, independently per trial."""
    return rng.integers(0, n, size=(n_trials, n))


def _block_bootstrap_matrix(
    n: int, block_length: int, n_trials: int, rng: np.random.Generator
) -> np.ndarray:
    """Resample contiguous blocks of `block_length` bars with replacement
    and concatenate them into a series of length n, preserving
    within-block autocorrelation. block_length<=1 has no blocks to
    preserve, so it is delegated to the simple iid bootstrap outright
    (rather than merely producing an equivalent result via block math),
    guaranteeing the two agree exactly for the same seed.
    """
    if block_length <= 1:
        return _simple_bootstrap_matrix(n, n_trials, rng)

    n_blocks = -(-n // block_length)  # ceil(n / block_length)
    block_starts = rng.integers(0, n - block_length + 1, size=(n_trials, n_blocks))
    offsets = np.arange(block_length)
    indices = block_starts[:, :, None] + offsets[None, None, :]
    return indices.reshape(n_trials, n_blocks * block_length)[:, :n]


def _perturb_prices(df: pd.DataFrame, noise_std: float, rng: np.random.Generator) -> pd.DataFrame:
    """Multiply OHLC by one shared per-bar Gaussian noise factor, so the
    perturbed series keeps consistent open/high/low/close relationships
    for any downstream strategy or indicator. Volume is left untouched."""
    noise_factor = 1 + rng.normal(loc=0.0, scale=noise_std, size=len(df))
    out = df.copy()
    for column in ("open", "high", "low", "close"):
        out[column] = out[column] * noise_factor
    return out


class MonteCarloAnalyzer:

    def __init__(
        self,
        n_trials: int = 5000,
        seed: int | None = 42,
        block_length: int = 20,
        noise_std: float = 0.001,
    ) -> None:
        self.n_trials = n_trials
        self.seed = seed
        self.block_length = block_length
        self.noise_std = noise_std

    def bootstrap(
        self, result: BacktestResult, block_length: int | None = None
    ) -> MonteCarloResult:
        """Resample `result`'s daily strategy returns (with replacement,
        same length) to rebuild an equity curve per trial, producing
        distributions of final return and max drawdown rather than the
        single numbers the historical backtest happened to produce."""
        block_length = self.block_length if block_length is None else block_length
        returns = result.strategy_returns.to_numpy()
        n = len(returns)
        starting_equity = float(result.equity_curve.iloc[0])

        rng = np.random.default_rng(self.seed)
        indices = _block_bootstrap_matrix(n, block_length, self.n_trials, rng)
        sampled_returns = returns[indices]

        equity_paths = starting_equity * np.cumprod(1 + sampled_returns, axis=1)
        final_equity = equity_paths[:, -1]
        running_max = np.maximum.accumulate(equity_paths, axis=1)
        max_drawdown = (equity_paths / running_max - 1).min(axis=1)

        actual_running_max = result.equity_curve.cummax()
        actual_max_drawdown = (result.equity_curve / actual_running_max - 1).min()

        return MonteCarloResult(
            final_equity=final_equity,
            max_drawdown=max_drawdown,
            actual_final_equity=float(result.equity_curve.iloc[-1]),
            actual_max_drawdown=float(actual_max_drawdown),
            initial_capital=starting_equity,
            block_length=block_length,
        )

    def noise_robustness(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        backtester: Backtester,
        noise_std: float | None = None,
    ) -> NoiseRobustnessResult:
        """Perturb the input price series with small Gaussian noise and
        re-run the full strategy + backtest on each perturbed series. This
        re-runs strategy.generate_signals and Backtester.run per trial (not
        a vectorised operation, unlike bootstrap) because the perturbation
        happens upstream of every indicator the strategy computes - the
        cost is the price of testing the strategy's actual computation
        path, the same trade-off WalkForwardValidator makes per fold."""
        noise_std = self.noise_std if noise_std is None else noise_std
        rng = np.random.default_rng(self.seed)

        final_equity = np.empty(self.n_trials)
        for trial in range(self.n_trials):
            perturbed = _perturb_prices(df, noise_std, rng)
            trial_result = backtester.run(perturbed, strategy)
            final_equity[trial] = trial_result.equity_curve.iloc[-1]

        actual_result = backtester.run(df, strategy)

        return NoiseRobustnessResult(
            final_equity=final_equity,
            actual_final_equity=float(actual_result.equity_curve.iloc[-1]),
            noise_std=noise_std,
        )

    def print_bootstrap_report(self, mc: MonteCarloResult, label: str) -> str:
        lines = [
            f"\nMonte Carlo bootstrap ({label}, block_length={mc.block_length}):",
            f"  actual final equity:   {mc.actual_final_equity:,.2f}",
            f"  actual max drawdown:   {mc.actual_max_drawdown:.4f}",
            f"  P(final equity < initial capital): {mc.prob_below_initial_capital():.4f}",
            "  final equity percentiles: "
            + ", ".join(f"p{p}={v:,.2f}" for p, v in mc.final_equity_percentiles().items()),
            "  max drawdown percentiles: "
            + ", ".join(f"p{p}={v:.4f}" for p, v in mc.max_drawdown_percentiles().items()),
        ]
        report = "\n".join(lines)
        print(report)
        return report

    def print_noise_report(self, nr: NoiseRobustnessResult, label: str) -> str:
        lines = [
            f"\nNoise-perturbation robustness ({label}, noise_std={nr.noise_std}):",
            f"  actual final equity: {nr.actual_final_equity:,.2f}",
            "  final equity percentiles: "
            + ", ".join(f"p{p}={v:,.2f}" for p, v in nr.final_equity_percentiles().items()),
        ]
        report = "\n".join(lines)
        print(report)
        return report
