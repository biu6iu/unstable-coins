from __future__ import annotations

from typing import Callable

from src.strategies.base import Strategy
from src.strategies.buy_and_hold import BuyAndHoldStrategy
from src.strategies.ensemble import SoftVotingStrategy
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.rsi_filter import RSIFilterStrategy
from src.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategies.tsmom import TimeSeriesMomentumStrategy
from src.strategies.volatility_breakout import DonchianBreakoutStrategy


def _build_rsi_filtered_crossover(
    fast: int = 20,
    slow: int = 50,
    rsi_window: int = 14,
    rsi_overbought: float = 70.0,
) -> RSIFilterStrategy:
    """RSIFilterStrategy wraps MACrossoverStrategy so the registry can build it from flat config params"""
    return RSIFilterStrategy(
        MACrossoverStrategy(fast=fast, slow=slow),
        rsi_window=rsi_window,
        rsi_overbought=rsi_overbought,
    )


def _build_soft_voting_ensemble(
    fast: int = 20,
    slow: int = 50,
    buy_below: float = 30.0,
    exit_above: float = 50.0,
    entry_window: int = 20,
    exit_window: int = 10,
    weights: list[float] | None = None,
) -> SoftVotingStrategy:
    """Soft-voting counterpart to the hard-voting ensemble in scripts/run_backtest.py, same three members"""
    return SoftVotingStrategy(
        [
            MACrossoverStrategy(fast=fast, slow=slow),
            RSIMeanReversionStrategy(buy_below=buy_below, exit_above=exit_above),
            DonchianBreakoutStrategy(entry_window=entry_window, exit_window=exit_window),
        ],
        weights=weights,
    )


STRATEGY_REGISTRY: dict[str, Callable[..., Strategy]] = {
    "buy_and_hold": BuyAndHoldStrategy,
    "ma_crossover": MACrossoverStrategy,
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "tsmom": TimeSeriesMomentumStrategy,
    "rsi_filtered_crossover": _build_rsi_filtered_crossover,
    "soft_voting_ensemble": _build_soft_voting_ensemble,
}


def build_strategies(strategy_configs: list[dict]) -> list[Strategy]:
    """Instantiate the strategies"""
    return [
        STRATEGY_REGISTRY[entry["name"]](**entry.get("params", {}))
        for entry in strategy_configs
    ]
