from abc import ABC, abstractmethod

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataProvider(ABC):
    """Base class for anything that can supply OHLCV candle data"""

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """return OHLCV data in the standard schema"""
        raise NotImplementedError


def validate(df: pd.DataFrame) -> None:
    """Raise if data doesn't conform to the standard OHLCV schema"""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected a DatetimeIndex, got {type(df.index).__name__}")
    if df.index.name != "timestamp":
        raise ValueError(f"Expected index name 'timestamp', got {df.index.name!r}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
