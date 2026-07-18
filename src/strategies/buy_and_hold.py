from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy


class BuyAndHoldStrategy(Strategy):

    @property
    def name(self) -> str:
        return "BuyAndHold"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = 1
        return out
