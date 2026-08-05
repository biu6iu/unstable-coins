from __future__ import annotations

import pandas as pd

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

    @property
    def name(self) -> str:
        return f"DonchianBreakout({self.entry_window},{self.exit_window})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = out["close"]

        entry_high = close.rolling(window=self.entry_window).max().shift(1)
        exit_low = close.rolling(window=self.exit_window).min().shift(1)

        entries = close > entry_high
        exits = close < exit_low

        out["signal"] = trigger_hold_signal(out.index, entries, exits)
        return out
