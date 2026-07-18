from abc import ABC, abstractmethod

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
