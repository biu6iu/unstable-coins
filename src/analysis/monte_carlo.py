from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.backtest.engine import Backtester
from src.backtest.result import BacktestResult
from src.evaluation.metrics import max_drawdown
from src.strategies.base import Strategy

PERCENTILES = (5, 25, 50, 75, 95)

def _percentile_summary(values: np.ndarray) -> dict[int, float]:
    return {p: float(np.percentile(values, p)) for p in PERCENTILES}


@dataclass
class MonteCarloResult:
    """
    Distributions of final equity and max drawdown from resampled trials of one strategy's daily returns, 
    compared with historical values.
    """

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
    """
    spread of final-equity outcomes when the same strategy is re-run on
    many small Gaussian perturbations of the input price series
    """

    final_equity: np.ndarray
    actual_final_equity: float
    noise_std: float

    def final_equity_percentiles(self) -> dict[int, float]:
        return _percentile_summary(self.final_equity)


def _simple_bootstrap_matrix(n: int, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """iid resampling"""
    return rng.integers(0, n, size=(n_trials, n))


def _block_bootstrap_matrix(n: int, block_length: int, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    if block_length <= 1:
        return _simple_bootstrap_matrix(n, n_trials, rng)

    # ceil(n / block_length)
    n_blocks = -(-n // block_length)  
    block_starts = rng.integers(0, n - block_length + 1, size=(n_trials, n_blocks))
    offsets = np.arange(block_length)
    indices = block_starts[:, :, None] + offsets[None, None, :]
    return indices.reshape(n_trials, n_blocks * block_length)[:, :n]


def _perturb_prices(df: pd.DataFrame, noise_std: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    Multiply OHLC by one shared per-bar Gaussian noise factor, so the
    perturbed series keeps consistent OHLC relationships
    for any downstream strategy or indicator
    """
    noise_factor = 1 + rng.normal(loc=0.0, scale=noise_std, size=len(df))
    out = df.copy()
    for column in ("open", "high", "low", "close"):
        out[column] = out[column] * noise_factor
    return out


class MonteCarloAnalyzer:

    def __init__(self,
        n_trials: int = 5000,
        seed: int | None = 42,
        block_length: int = 20,
        noise_std: float = 0.001,
    ) -> None:
        
        self.n_trials = n_trials
        self.seed = seed
        self.block_length = block_length
        self.noise_std = noise_std

    def bootstrap(self, result: BacktestResult, block_length: int | None = None) -> MonteCarloResult:
        """Resample `result`'s daily strategy returns to rebuild an equity curve per trial"""
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
        trial_max_drawdown = (equity_paths / running_max - 1).min(axis=1)

        actual_max_drawdown = max_drawdown(result.equity_curve)

        return MonteCarloResult(
            final_equity=final_equity,
            max_drawdown=trial_max_drawdown,
            actual_final_equity=float(result.equity_curve.iloc[-1]),
            actual_max_drawdown=float(actual_max_drawdown),
            initial_capital=starting_equity,
            block_length=block_length
        )

    def noise_robustness(self, df: pd.DataFrame, strategy: Strategy, backtester: Backtester, noise_std: float | None = None) -> NoiseRobustnessResult:
        """
        perturb the input price series with small Gaussian noise and
        re-run the full strategy + backtest on each perturbed series
        """
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

    def compare_ensemble_drawdowns(self, ensemble_result: BacktestResult, member_results: list[BacktestResult]) -> dict[str, MonteCarloResult]:
        comparisons = {ensemble_result.strategy_name: self.bootstrap(ensemble_result)}
        for member_result in member_results:
            comparisons[member_result.strategy_name] = self.bootstrap(member_result)
        return comparisons

    def print_drawdown_comparison(self, comparisons: dict[str, MonteCarloResult]) -> str:
        lines = ["\nMonte Carlo max-drawdown comparison (ensemble vs members):"]
        for name, mc in comparisons.items():
            percentiles = mc.max_drawdown_percentiles()
            lines.append(
                f"  {name}: actual={mc.actual_max_drawdown:.4f}  "
                + ", ".join(f"p{p}={v:.4f}" for p, v in percentiles.items())
            )
        
        report = "\n".join(lines)
        print(report)
        return report

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
