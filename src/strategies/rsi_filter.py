from __future__ import annotations
import pandas as pd

from src.preprocessing.features import FeatureEngineer
from src.strategies.base import Strategy


class RSIFilterStrategy(Strategy):
    """
    RSI above `rsi_overbought` (default 70) means the asset has already
    run up sharply, so entering a new long here is exactly when a
    trend-following signal is most likely to be chasing a local top. This
    filter only vetoes entries into an overbought market, it does not
    change the wrapped strategy's exit logic.
    """

    def __init__(
        self,
        strategy: Strategy,
        rsi_window: int = 14,
        rsi_overbought: float = 70.0,
    ) -> None:
        self.strategy = strategy
        self.rsi_window = rsi_window
        self.rsi_overbought = rsi_overbought
        self._engineer = FeatureEngineer()

    @property
    def name(self) -> str:
        return f"RSIFilter({self.strategy.name},{self.rsi_overbought})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.strategy.generate_signals(df)
        out = self._engineer.rsi(out, window=self.rsi_window)

        overbought = out[f"rsi_{self.rsi_window}"] > self.rsi_overbought
        out.loc[overbought, "signal"] = 0
        return out
