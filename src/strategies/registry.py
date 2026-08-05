from __future__ import annotations

from src.strategies.base import Strategy
from src.strategies.buy_and_hold import BuyAndHoldStrategy
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from src.strategies.tsmom import TimeSeriesMomentumStrategy
from src.strategies.volatility_breakout import DonchianBreakoutStrategy

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "buy_and_hold": BuyAndHoldStrategy,
    "ma_crossover": MACrossoverStrategy,
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "tsmom": TimeSeriesMomentumStrategy,
}


def build_strategies(strategy_configs: list[dict]) -> list[Strategy]:
    """Instantiate the strategies"""
    return [
        STRATEGY_REGISTRY[entry["name"]](**entry.get("params", {}))
        for entry in strategy_configs
    ]
