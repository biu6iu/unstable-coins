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
        since: pd.Timestamp | None = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.exchange_id = exchange_id
        self.cache_dir = Path(cache_dir)

        # a `since`-bounded fetch targets a specific historical range which the plain symbol/timeframe/limit cache key doesn't capture
        self.use_cache = use_cache and since is None
        self.since = since

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
        since_ms = int(self.since.timestamp() * 1000) if self.since is not None else None
        raw = exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, since=since_ms, limit=self.limit)

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp").astype(float)

        validate(df)

        # save the data so the next fetch() is a cache hit
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)

        return df
