from __future__ import annotations

import pandas as pd

from src.backtest.result import BacktestResult
from src.strategies.base import Strategy


class VotingStrategy(Strategy):
    """
    Hard voting: signal is 1 when at least `k` of `len(strategies)` members are long, else 0
    """

    def __init__(self, strategies: list[Strategy], k: int) -> None:
        if not 1 <= k <= len(strategies):
            raise ValueError(
                f"k ({k}) must be between 1 and ({len(strategies)})"
            )
        self.strategies = strategies
        self.k = k

    @property
    def name(self) -> str:
        members = ",".join(s.name for s in self.strategies)
        return f"Voting({members};k={self.k})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        member_signals = [s.generate_signals(df)["signal"] for s in self.strategies]
        votes = sum(member_signals)

        out = df.copy()
        out["signal"] = (votes >= self.k).astype(int)
        return out


class SoftVotingStrategy(Strategy):
    """
    Soft voting: signal is the (optionally weighted) mean of member signals.
    
    The signal is interpreted as conviction-based position sizing rather than a binary decision
    """

    def __init__(self, strategies: list[Strategy], weights: list[float] | None = None) -> None:
        if weights is not None and len(weights) != len(strategies):
            raise ValueError("weights must have the same length as strategies")
        self.strategies = strategies
        self.weights = weights

    @property
    def name(self) -> str:
        members = ",".join(s.name for s in self.strategies)
        return f"SoftVoting({members})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        weights = self.weights if self.weights is not None else [1.0] * len(self.strategies)
        member_signals = [s.generate_signals(df)["signal"] for s in self.strategies]
        weighted_sum = sum(signal * weight for signal, weight in zip(member_signals, weights))

        out = df.copy()
        out["signal"] = weighted_sum / sum(weights)
        return out


def correlation_matrix(results: list[BacktestResult]) -> pd.DataFrame:
    """pairwise daily-return correlation matrix across member backtests"""
    returns = pd.concat(
        {result.strategy_name: result.strategy_returns for result in results}, axis=1
    )
    return returns.corr()


def print_correlation_matrix(results: list[BacktestResult]) -> str:
    matrix = correlation_matrix(results)
    report = ("\nPairwise daily-return correlation (ensemble members):\n" 
              + matrix.to_string(float_format=lambda v: f"{v:.3f}"))
    print(report)
    return report
