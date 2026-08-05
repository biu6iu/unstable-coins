from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Strategy(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `df` with an added `signal` column. `signal` is 1 (long) or 0 (flat) for each bar."""
        raise NotImplementedError


def trigger_hold_signal(index: pd.Index, enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """
    Mark entry/exit trigger bars 1/0 and forward-fill so the position holds
    between them. Shared by strategies whose signal is an entry/exit
    condition (e.g. RSI mean reversion, Donchian breakout) rather than a
    continuously-recomputed rule (e.g. MA crossover).
    """
    raw = pd.Series(np.nan, index=index)
    raw[enter] = 1.0
    raw[exit_] = 0.0
    return raw.ffill().fillna(0.0).astype(int)
