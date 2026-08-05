from __future__ import annotations

import pandas as pd

from src.preprocessing.features import FeatureEngineer
from src.strategies.base import Strategy, trigger_hold_signal


class DonchianBreakoutStrategy(Strategy):
    """
    goes long when the close exceeds the prior `entry_window`-bar high, and exits when the close falls below the prior
    `exit_window`-bar low.
    `exit_window` is conventionally shorter than `entry_window` (exit faster than entry) to cut losing breakouts quickly
    while giving winners room to run.
    """

    def __init__(self, entry_window: int = 20, exit_window: int = 10) -> None:
        self.entry_window = entry_window
        self.exit_window = exit_window
        self._engineer = FeatureEngineer()

    @property
    def name(self) -> str:
        return f"DonchianBreakout({self.entry_window},{self.exit_window})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._engineer.rolling_max(df, window=self.entry_window)
        out = self._engineer.rolling_min(out, window=self.exit_window)
        close = out["close"]

        # shift so today's entry/exit decision only sees the channel formed by PRIOR bars
        entry_high = out[f"rolling_max_{self.entry_window}"].shift(1)
        exit_low = out[f"rolling_min_{self.exit_window}"].shift(1)

        entries = close > entry_high
        exits = close < exit_low

        out["signal"] = trigger_hold_signal(out.index, entries, exits)
        return out
