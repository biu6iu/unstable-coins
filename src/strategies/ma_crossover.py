from __future__ import annotations

import pandas as pd

from src.preprocessing.features import FeatureEngineer
from src.strategies.base import Strategy


class MACrossoverStrategy(Strategy):
    """Goes long when the fast SMA is above the slow SMA"""

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be less than slow ({slow})")
        self.fast = fast
        self.slow = slow
        self._engineer = FeatureEngineer()

    @property
    def name(self) -> str:
        return f"MACrossover({self.fast},{self.slow})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._engineer.sma(df, window=self.fast)
        out = self._engineer.sma(out, window=self.slow)

        fast_col = f"sma_{self.fast}"
        slow_col = f"sma_{self.slow}"
        
        out["signal"] = (out[fast_col] > out[slow_col]).astype(int)
        return out
