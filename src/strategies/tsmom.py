from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy


class TimeSeriesMomentumStrategy(Strategy):
    """
    long whenever the close is above its own value `lookback` bars ago, flat otherwise.
    """

    def __init__(self, lookback: int = 90) -> None:
        self.lookback = lookback

    @property
    def name(self) -> str:
        return f"TSMOM({self.lookback})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = (out["close"] > out["close"].shift(self.lookback)).astype(int)
        return out
