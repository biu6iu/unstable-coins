from __future__ import annotations
from pathlib import Path
from typing import Any
import ccxt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.base import DataProvider, validate

DEFAULT_CACHE_DIR = Path("data/raw")

# parquet metadata keys recording which window a cache file was fetched for
COVERED_START_KEY = b"covered_start_ms"
COVERED_END_KEY = b"covered_end_ms"

# exchanges cap a single fetch_ohlcv call (binance: 1000 bars), which is what
# limited history to ~2 years before pagination existed
DEFAULT_BATCH_SIZE = 1000

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


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
        batch_size: int = DEFAULT_BATCH_SIZE,
        exchange: Any | None = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.exchange_id = exchange_id
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.since = since
        self.batch_size = batch_size
        self._exchange = exchange

    def _get_exchange(self) -> Any:
        if self._exchange is None:
            exchange_class = getattr(ccxt, self.exchange_id)
            self._exchange = exchange_class({"enableRateLimit": True})
        return self._exchange

    def _cache_path(self) -> Path:
        safe_symbol = self.symbol.replace("/", "-")
        return self.cache_dir / f"{self.exchange_id}_{safe_symbol}_{self.timeframe}.parquet"

    def _request_window(self, exchange: Any) -> tuple[int, int, int]:
        """resolve the wanted (first bar, last bar, bar length) in epoch ms"""
        tf_ms = exchange.parse_timeframe(self.timeframe) * 1000
        if self.since is not None:
            start_ms = int(self.since.timestamp() * 1000)
            end_ms = start_ms + (self.limit - 1) * tf_ms

        # the bar currently forming is excluded, so the newest bar we want opened one timeframe ago
        else:
            end_ms = exchange.milliseconds() - tf_ms
            start_ms = end_ms - self.limit * tf_ms
        return start_ms, end_ms, tf_ms

    def fetch(self) -> pd.DataFrame:
        exchange = self._get_exchange()
        start_ms, end_ms, tf_ms = self._request_window(exchange)

        if self.use_cache:
            cached = self._read_cache(start_ms, end_ms, tf_ms)
            if cached is not None:
                validate(cached)
                return cached

        rows = self._paginate(exchange, start_ms, end_ms, tf_ms)
        df = self._to_frame(rows, exchange.milliseconds(), tf_ms)
        df = self._trim(df, start_ms)

        validate(df)

        if self.use_cache:
            self._write_cache(df, start_ms, end_ms, tf_ms)

        return df

    def _paginate(self, exchange: Any, start_ms: int, end_ms: int, tf_ms: int) -> list[list]:
        """walk `since` forward one batch at a time until the window is covered"""
        rows: list[list] = []
        cursor = start_ms
        while cursor <= end_ms:
            batch = exchange.fetch_ohlcv(
                self.symbol, timeframe=self.timeframe, since=cursor, limit=self.batch_size
            )
            if not batch:
                break
            rows.extend(batch)
            next_cursor = batch[-1][0] + tf_ms
            if next_cursor <= cursor or len(batch) < self.batch_size:
                break
            cursor = next_cursor
        return rows

    def _to_frame(self, rows: list[list], now_ms: int, tf_ms: int) -> pd.DataFrame:
        """stitch the batches into the standard schema"""
        if not rows:
            raise ValueError(f"exchange returned no OHLCV data for {self.symbol} {self.timeframe}")

        df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
        df = df.drop_duplicates(subset="timestamp", keep="first")
        df = df[df["timestamp"] + tf_ms <= now_ms]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.set_index("timestamp").sort_index().astype(float)

    def _trim(self, df: pd.DataFrame, start_ms: int) -> pd.DataFrame:
        """cut the stitched history down to exactly what was asked for"""
        df = df[df.index >= pd.to_datetime(start_ms, unit="ms")]
        return df.head(self.limit) if self.since is not None else df.tail(self.limit)

    def _read_cache(self, start_ms: int, end_ms: int, tf_ms: int) -> pd.DataFrame | None:
        path = self._cache_path()
        if not path.exists():
            return None
        
        table = pq.read_table(path)
        cached = table.to_pandas()
        if cached.empty:
            return None
    
        covered_start, covered_end = self._covered_range(table, cached)
        if covered_start > start_ms or covered_end < end_ms - tf_ms:
            return None
        return self._trim(cached, start_ms)

    def _write_cache(self, df: pd.DataFrame, start_ms: int, end_ms: int, tf_ms: int) -> None:
        """merge into the market's cache file, keeping its coverage contiguous"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path()
        merged, covered = df, (start_ms, end_ms)
        if path.exists():
            table = pq.read_table(path)
            cached = table.to_pandas()
            previous = self._covered_range(table, cached)
            if not cached.empty and self._ranges_join(previous, covered, tf_ms):
                merged = pd.concat([cached, df])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                covered = (min(previous[0], start_ms), max(previous[1], end_ms))
        self._write_parquet(merged, covered, path)

    @staticmethod
    def _write_parquet(df: pd.DataFrame, covered: tuple[int, int], path: Path) -> None:
        table = pa.Table.from_pandas(df)
        metadata = {
            **(table.schema.metadata or {}),
            COVERED_START_KEY: str(covered[0]).encode(),
            COVERED_END_KEY: str(covered[1]).encode(),
        }
        pq.write_table(table.replace_schema_metadata(metadata), path)

    @classmethod
    def _covered_range(cls, table: "pa.Table", df: pd.DataFrame) -> tuple[int, int]:
        metadata = table.schema.metadata or {}
        if COVERED_START_KEY in metadata and COVERED_END_KEY in metadata:
            return int(metadata[COVERED_START_KEY]), int(metadata[COVERED_END_KEY])
        return cls._bounds_ms(df)

    @staticmethod
    def _bounds_ms(df: pd.DataFrame) -> tuple[int, int]:
        return df.index[0].value // 1_000_000, df.index[-1].value // 1_000_000

    @staticmethod
    def _ranges_join(a: tuple[int, int], b: tuple[int, int], tf_ms: int) -> bool:
        """true when the two covered ranges overlap or sit one bar apart"""
        return a[0] <= b[1] + tf_ms and b[0] <= a[1] + tf_ms
