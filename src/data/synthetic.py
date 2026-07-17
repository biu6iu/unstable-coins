"""Synthetic OHLCV data generator for tests and offline development."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.base import DataProvider, validate


class SyntheticDataProvider(DataProvider):
    """Generates OHLCV candles via GBM with alternating bull/bear regimes."""

    def __init__(
        self,
        n_periods: int = 730,
        start: str = "2022-01-01",
        freq: str = "D",
        start_price: float = 100.0,
        regime_length: int = 90,
        bull_drift: float = 0.0015,
        bear_drift: float = -0.0015,
        volatility: float = 0.02,
        seed: int | None = 42,
    ) -> None:
        self.n_periods = n_periods
        self.start = start
        self.freq = freq
        self.start_price = start_price
        self.regime_length = regime_length
        self.bull_drift = bull_drift
        self.bear_drift = bear_drift
        self.volatility = volatility
        self.seed = seed

    def fetch(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)

        # Alternate bull/bear drift every `regime_length` bars so the
        # series actually trends and reverses instead of drifting flatly.
        regime_index = np.arange(self.n_periods) // self.regime_length
        drift = np.where(regime_index % 2 == 0, self.bull_drift, self.bear_drift)

        shocks = rng.normal(loc=0.0, scale=self.volatility, size=self.n_periods)
        close = self.start_price * np.exp(np.cumsum(drift + shocks))

        # Open is the prior close; high/low add small synthetic wicks around open/close.
        open_ = np.empty_like(close)
        open_[0] = self.start_price
        open_[1:] = close[:-1]
        wick = np.abs(rng.normal(loc=0.0, scale=self.volatility / 2, size=self.n_periods))
        high = np.maximum(open_, close) * (1 + wick)
        low = np.minimum(open_, close) * (1 - wick)
        volume = rng.uniform(low=100.0, high=1000.0, size=self.n_periods)

        index = pd.date_range(
            start=self.start, periods=self.n_periods, freq=self.freq, name="timestamp"
        )
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        ).astype(float)

        validate(df)
        return df
