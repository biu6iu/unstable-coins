"""Live OHLCV data from a real exchange via ccxt, with local caching"""

from __future__ import annotations
from pathlib import Path

import ccxt
import pandas as pd

from src.data.base import DataProvider, validate

DEFAULT_CACHE_DIR = Path("data/raw")

class CCXTDataProvider(DataProvider):
    """Fetches OHLCV candles from a ccxt exchange"""

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1d",
        limit: int = 730, # ~2 years of daily candles
        exchange_id: str = "binance",
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        use_cache: bool = True,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.exchange_id = exchange_id
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

    def _cache_path(self) -> Path:
        safe_symbol = self.symbol.replace("/", "-")
        filename = f"{self.exchange_id}_{safe_symbol}_{self.timeframe}_{self.limit}.parquet"
        return self.cache_dir / filename

    def fetch(self) -> pd.DataFrame:
        cache_path = self._cache_path()
        # avoid refetching repeated runs
        if self.use_cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            validate(df)
            return df

        # fetch data and transform into a valid df 
        exchange_class = getattr(ccxt, self.exchange_id)
        exchange = exchange_class() 
        raw = exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=self.limit)

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp").astype(float)

        validate(df)

        # save the data so the next fetch() is a cache hit
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)

        return df
