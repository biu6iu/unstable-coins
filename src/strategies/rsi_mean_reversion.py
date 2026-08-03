from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.features import FeatureEngineer
from src.strategies.base import Strategy


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

        # mark only the entry/exit trigger bars, then forward-fill so the position holds between them
        raw = pd.Series(np.nan, index=out.index)
        raw[rsi < self.buy_below] = 1.0
        raw[rsi > self.exit_above] = 0.0
        out["signal"] = raw.ffill().fillna(0.0).astype(int)
        return out
