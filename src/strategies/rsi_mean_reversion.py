from __future__ import annotations

import pandas as pd

from src.preprocessing.features import FeatureEngineer
from src.strategies.base import Strategy, trigger_hold_signal


class RSIMeanReversionStrategy(Strategy):
    """
    Goes long when RSI drops below `buy_below` (oversold) and holds the position until RSI recovers above `exit_above`
    """

    def __init__(self, window: int = 14, buy_below: float = 30.0, exit_above: float = 50.0) -> None:
        if buy_below >= exit_above:
            raise ValueError(f"buy_below ({buy_below}) must be less than exit_above ({exit_above})")
        self.window = window
        self.buy_below = buy_below
        self.exit_above = exit_above
        self._engineer = FeatureEngineer()

    @property
    def name(self) -> str:
        return f"RSIMeanReversion({self.window},{self.buy_below},{self.exit_above})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._engineer.rsi(df, window=self.window)
        rsi = out[f"rsi_{self.window}"]

        out["signal"] = trigger_hold_signal(out.index, rsi < self.buy_below, rsi > self.exit_above)
        return out
