"""Tests for the data layer."""

import pandas as pd
import pytest

from src.data.base import validate
from src.data.ccxt_provider import CCXTDataProvider
from src.data.synthetic import SyntheticDataProvider

DAY_MS = 86_400_000
FIRST_BAR_MS = int(pd.Timestamp("2020-01-01").timestamp() * 1000)


def make_bars(n: int, first_ms: int = FIRST_BAR_MS) -> list[list[float]]:
    """OHLCV rows in ccxt's raw list format, one bar per day"""
    return [
        [first_ms + i * DAY_MS, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0 + i]
        for i in range(n)
    ]


class FakeExchange:
    """Scripted stand-in for a ccxt exchange.

    Serves bars from an in-memory history so pagination can be exercised
    without the network, and records every `since` it was called with so the
    paging sequence itself can be asserted on.
    """

    def __init__(self, bars, now_ms: int, per_request: int = 1000, seam_overlap: int = 0) -> None:
        self.bars = bars
        self.now_ms = now_ms
        self.per_request = per_request
        # real exchanges include the `since` bar itself, so consecutive batches
        # can overlap at the seam; this reproduces that
        self.seam_overlap = seam_overlap
        self.calls: list[int] = []

    def parse_timeframe(self, timeframe: str) -> int:
        return {"1d": 86_400, "1h": 3_600}[timeframe]

    def milliseconds(self) -> int:
        return self.now_ms

    def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
        self.calls.append(since)
        floor_ms = since - self.seam_overlap * DAY_MS
        available = [list(bar) for bar in self.bars if bar[0] >= floor_ms]
        return available[: min(limit or self.per_request, self.per_request)]


class OfflineExchange(FakeExchange):
    """Fails any actual fetch, so a test can prove a result came from cache"""

    def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
        raise AssertionError("fetch_ohlcv called when the cache should have served the request")


def test_synthetic_provider_returns_standard_schema():
    df = SyntheticDataProvider(n_periods=100, seed=1).fetch()

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "timestamp"
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 100
    assert df.index.is_monotonic_increasing
    assert (df.dtypes == float).all()


def test_synthetic_provider_is_reproducible_with_seed():
    df1 = SyntheticDataProvider(n_periods=50, seed=7).fetch()
    df2 = SyntheticDataProvider(n_periods=50, seed=7).fetch()

    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_provider_different_seeds_differ():
    df1 = SyntheticDataProvider(n_periods=50, seed=1).fetch()
    df2 = SyntheticDataProvider(n_periods=50, seed=2).fetch()

    assert not df1["close"].equals(df2["close"])


def test_validate_ohlcv_schema_rejects_non_datetime_index():
    df = pd.DataFrame(
        {"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}
    )
    with pytest.raises(TypeError):
        validate(df)


def test_validate_ohlcv_schema_rejects_wrong_index_name():
    index = pd.date_range("2022-01-01", periods=2, name="date")
    df = pd.DataFrame(
        {"open": [1, 1], "high": [1, 1], "low": [1, 1], "close": [1, 1], "volume": [1, 1]},
        index=index,
    )
    with pytest.raises(ValueError):
        validate(df)


def test_validate_ohlcv_schema_rejects_missing_columns():
    index = pd.date_range("2022-01-01", periods=2, name="timestamp")
    df = pd.DataFrame({"open": [1, 1], "close": [1, 1]}, index=index)
    with pytest.raises(ValueError):
        validate(df)


def build_ccxt_provider(exchange, limit, tmp_path=None, **kwargs) -> CCXTDataProvider:
    return CCXTDataProvider(
        limit=limit,
        exchange=exchange,
        use_cache=tmp_path is not None,
        cache_dir=tmp_path or ".",
        **kwargs,
    )


def test_ccxt_provider_stitches_paginated_batches_in_order():
    bars = make_bars(11)
    # the newest bar is mid-formation, so ten closed bars are on offer
    exchange = FakeExchange(bars, now_ms=bars[-1][0] + DAY_MS // 2)
    provider = build_ccxt_provider(exchange, limit=10, batch_size=3)

    df = provider.fetch()

    assert len(exchange.calls) > 1, "history longer than one batch should page"
    assert len(df) == 10
    assert df.index.is_monotonic_increasing
    assert df.index[0] == pd.Timestamp("2020-01-01")
    assert list(df["close"]) == [bar[4] for bar in bars[:10]]


def test_ccxt_provider_drops_duplicate_bars_at_batch_seams():
    bars = make_bars(11)
    exchange = FakeExchange(bars, now_ms=bars[-1][0] + DAY_MS // 2, seam_overlap=1)
    provider = build_ccxt_provider(exchange, limit=10, batch_size=3)

    df = provider.fetch()

    assert df.index.is_unique
    assert len(df) == 10


def test_ccxt_provider_drops_the_in_progress_final_candle():
    bars = make_bars(6)
    # halfway through the last bar: its close is not final yet
    exchange = FakeExchange(bars, now_ms=bars[-1][0] + DAY_MS // 2)
    provider = build_ccxt_provider(exchange, limit=10)

    df = provider.fetch()

    assert len(df) == 5
    assert df.index[-1] == pd.Timestamp(bars[-2][0], unit="ms")


def test_ccxt_provider_stops_paging_on_a_short_final_batch():
    bars = make_bars(5)
    exchange = FakeExchange(bars, now_ms=bars[-1][0] + 2 * DAY_MS)
    # asks for far more history than exists; a short batch must end the loop
    provider = build_ccxt_provider(exchange, limit=500, batch_size=10)

    df = provider.fetch()

    assert len(exchange.calls) == 1
    assert len(df) == 5


def test_ccxt_provider_returns_standard_schema():
    bars = make_bars(20)
    exchange = FakeExchange(bars, now_ms=bars[-1][0] + DAY_MS // 2)
    provider = build_ccxt_provider(exchange, limit=15, batch_size=4)

    df = provider.fetch()

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "timestamp"
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert (df.dtypes == float).all()


def test_ccxt_provider_keeps_the_most_recent_bars_when_limit_is_short():
    bars = make_bars(20)
    exchange = FakeExchange(bars, now_ms=bars[-1][0] + DAY_MS // 2)
    provider = build_ccxt_provider(exchange, limit=5, batch_size=4)

    df = provider.fetch()

    assert len(df) == 5
    assert df.index[-1] == pd.Timestamp(bars[-2][0], unit="ms")


def test_ccxt_provider_since_takes_the_first_limit_bars_from_that_date():
    bars = make_bars(20)
    exchange = FakeExchange(bars, now_ms=bars[-1][0] + DAY_MS // 2)
    provider = build_ccxt_provider(
        exchange, limit=3, batch_size=2, since=pd.Timestamp("2020-01-05")
    )

    df = provider.fetch()

    assert list(df.index) == list(pd.date_range("2020-01-05", periods=3, name="timestamp"))


def test_ccxt_provider_raises_when_the_exchange_returns_nothing():
    exchange = FakeExchange([], now_ms=FIRST_BAR_MS)
    provider = build_ccxt_provider(exchange, limit=10)

    with pytest.raises(ValueError):
        provider.fetch()


def test_ccxt_provider_serves_a_request_the_cache_file_spans(tmp_path):
    bars = make_bars(20)
    now_ms = bars[-1][0] + DAY_MS // 2
    first = build_ccxt_provider(FakeExchange(bars, now_ms), limit=15, tmp_path=tmp_path)
    expected = first.fetch()

    # a narrower request over the same window must not touch the exchange again
    second = build_ccxt_provider(OfflineExchange(bars, now_ms), limit=10, tmp_path=tmp_path)
    df = second.fetch()

    assert len(df) == 10
    pd.testing.assert_frame_equal(df, expected.tail(10))


def test_ccxt_provider_refetches_when_the_request_predates_cached_coverage(tmp_path):
    bars = make_bars(20)
    now_ms = bars[-1][0] + DAY_MS // 2
    build_ccxt_provider(FakeExchange(bars, now_ms), limit=5, tmp_path=tmp_path).fetch()

    exchange = FakeExchange(bars, now_ms)
    df = build_ccxt_provider(exchange, limit=19, tmp_path=tmp_path).fetch()

    assert exchange.calls, "a request reaching further back than the cache must refetch"
    assert len(df) == 19
