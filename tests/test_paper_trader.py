import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Backtester
from src.data.base import DataProvider
from src.live.paper_trader import (
    DecisionRecord,
    PaperTrader,
    append_jsonl,
    diff_equity_paths,
    fetch_with_retry,
    load_decision_log,
    parse_timeframe_seconds,
)
from src.strategies.base import Strategy


def _make_ohlcv_at(timestamps, closes) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    index = pd.DatetimeIndex(timestamps, name="timestamp")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": np.full(len(closes), 100.0),
        },
        index=index,
    )


def _backtester() -> Backtester:
    return Backtester(fee=0.001, initial_capital=10000.0, slippage_bps=5.0)


class _ThresholdStrategy(Strategy):
    """Test double: long whenever close exceeds `threshold`. Recomputed on whatever-length df it's given, unlike a fixed signal list, so it works with PaperTrader's growing history."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return f"Threshold({self.threshold})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = (out["close"] > self.threshold).astype(int)
        return out


class _ScriptedProvider(DataProvider):
    """Test double: returns (or raises) the next scripted response on each fetch() call."""

    def __init__(self, responses: list, timeframe: str = "1h"):
        self._responses = list(responses)
        self.symbol = "BTC/USDT"
        self.timeframe = timeframe
        self.exchange_id = "binance"

    def fetch(self) -> pd.DataFrame:
        if not self._responses:
            raise RuntimeError("no more scripted responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


# --- fetch_with_retry ---------------------------------------------------


def test_fetch_with_retry_retries_then_succeeds():
    df = _make_ohlcv_at(pd.date_range("2024-01-01", periods=3, freq="h"), [1, 2, 3])
    provider = _ScriptedProvider([RuntimeError("boom"), RuntimeError("boom"), df])
    sleep_calls = []

    result = fetch_with_retry(provider, max_retries=5, base_delay_seconds=1.0, sleep_fn=sleep_calls.append)

    assert result is df
    assert sleep_calls == [1.0, 2.0]


def test_fetch_with_retry_returns_none_after_exhausting_retries():
    provider = _ScriptedProvider([RuntimeError("boom")] * 5)
    sleep_calls = []

    result = fetch_with_retry(provider, max_retries=5, base_delay_seconds=1.0, sleep_fn=sleep_calls.append)

    assert result is None
    assert len(sleep_calls) == 4  # no sleep after the final failed attempt


# --- PaperTrader.poll_once: new-bar detection & history accumulation ----


def test_poll_once_returns_none_when_no_new_closed_bar(tmp_path):
    ts = pd.date_range("2024-01-01T00:00", periods=4, freq="h")
    window = _make_ohlcv_at(ts, [90, 90, 90, 90])
    provider = _ScriptedProvider([window, window])
    trader = PaperTrader(
        provider=provider, strategy=_ThresholdStrategy(100), backtester=_backtester(),
        log_path=tmp_path / "log.jsonl", poll_interval_seconds=3600, sleep_fn=lambda s: None,
    )

    first = trader.poll_once()
    lines_after_first = (tmp_path / "log.jsonl").read_text().splitlines()
    second = trader.poll_once()
    lines_after_second = (tmp_path / "log.jsonl").read_text().splitlines()

    assert first is not None
    assert second is None
    assert lines_after_second == lines_after_first


def test_poll_once_ignores_still_forming_last_bar(tmp_path):
    ts = pd.date_range("2024-01-01T00:00", periods=4, freq="h")
    window = _make_ohlcv_at(ts, [90, 90, 90, 200])  # last bar's extreme close must not affect anything
    provider = _ScriptedProvider([window])
    trader = PaperTrader(
        provider=provider, strategy=_ThresholdStrategy(100), backtester=_backtester(),
        log_path=tmp_path / "log.jsonl", poll_interval_seconds=3600, sleep_fn=lambda s: None,
    )

    record = trader.poll_once()

    assert record.bar_timestamp == ts[2].isoformat()
    assert ts[3] not in trader._history.index


def test_poll_once_appends_exactly_the_newly_closed_bars(tmp_path):
    ts1 = pd.date_range("2024-01-01T00:00", periods=4, freq="h")
    window1 = _make_ohlcv_at(ts1, [90, 90, 90, 90])
    ts2 = pd.date_range("2024-01-01T01:00", periods=5, freq="h")  # overlaps, extends forward
    window2 = _make_ohlcv_at(ts2, [90, 90, 110, 110, 110])

    provider = _ScriptedProvider([window1, window2])
    trader = PaperTrader(
        provider=provider, strategy=_ThresholdStrategy(100), backtester=_backtester(),
        log_path=tmp_path / "log.jsonl", poll_interval_seconds=3600, sleep_fn=lambda s: None,
    )

    trader.poll_once()
    assert list(trader._history.index) == list(ts1[:3])

    trader.poll_once()
    assert list(trader._history.index) == list(ts1[:3]) + [ts2[2], ts2[3]]
    assert not trader._history.index.duplicated().any()


def test_gap_warning_logged_on_possible_missed_bars(tmp_path, caplog):
    ts1 = pd.date_range("2024-01-01T00:00", periods=4, freq="h")
    window1 = _make_ohlcv_at(ts1, [90, 90, 90, 90])
    ts2 = pd.date_range("2024-01-01T05:00", periods=4, freq="h")  # skips 03:00 and 04:00
    window2 = _make_ohlcv_at(ts2, [90, 90, 90, 90])

    provider = _ScriptedProvider([window1, window2])
    trader = PaperTrader(
        provider=provider, strategy=_ThresholdStrategy(100), backtester=_backtester(),
        log_path=tmp_path / "log.jsonl", poll_interval_seconds=3600, sleep_fn=lambda s: None,
    )

    with caplog.at_level(logging.WARNING):
        trader.poll_once()
        trader.poll_once()

    assert any("missed bars" in record.message for record in caplog.records)


def test_first_poll_seeds_full_warmup_history(tmp_path):
    window = _make_ohlcv_at(pd.date_range("2024-01-01T00:00", periods=4, freq="h"), [90, 90, 90, 110])
    provider = _ScriptedProvider([window])
    trader = PaperTrader(
        provider=provider, strategy=_ThresholdStrategy(100), backtester=_backtester(),
        log_path=tmp_path / "log.jsonl", poll_interval_seconds=3600, sleep_fn=lambda s: None,
    )

    trader.poll_once()

    assert len(trader._history) == len(window) - 1


# --- Portfolio state must never diverge from a direct Backtester.run() --


def test_portfolio_state_matches_direct_backtester_run(tmp_path):
    ts_all = pd.date_range("2024-01-01T00:00", periods=8, freq="h")
    closes = [90, 90, 90, 110, 110, 130, 90, 90]
    df_full = _make_ohlcv_at(ts_all, closes)
    strategy = _ThresholdStrategy(100)

    windows = [
        df_full.iloc[0:4],  # closed: idx0-2 -> new: idx0,1,2
        df_full.iloc[1:6],  # closed: idx1-4 -> new: idx3,4
        df_full.iloc[3:8],  # closed: idx3-6 -> new: idx5,6
    ]
    provider = _ScriptedProvider(windows)
    trader = PaperTrader(
        provider=provider, strategy=strategy, backtester=_backtester(),
        log_path=tmp_path / "log.jsonl", poll_interval_seconds=3600, sleep_fn=lambda s: None,
    )

    record1 = trader.poll_once()
    record2 = trader.poll_once()
    record3 = trader.poll_once()

    # position transitions from 0 -> 1 between idx3 and idx4 (poll 2), then holds through idx6 (poll 3)
    assert record1.is_trade is False
    assert record2.is_trade is True
    assert record3.is_trade is False

    assert list(trader._history.index) == list(df_full.index[:7])

    expected = _backtester().run(df_full.iloc[:7], strategy)
    assert record3.position == pytest.approx(expected.positions.iloc[-1])
    assert record3.equity == pytest.approx(expected.equity_curve.iloc[-1])
    assert record3.cash == pytest.approx(expected.cash.iloc[-1])
    assert record3.units == pytest.approx(expected.units.iloc[-1])
    assert record3.fee_drag == pytest.approx(expected.fee_drag.iloc[-1])
    assert record3.slippage_drag == pytest.approx(expected.slippage_drag.iloc[-1])
    assert record3.close == pytest.approx(expected.df["close"].iloc[-1])


# --- resilience: poll_once/run_forever never crash on provider failure --


def test_poll_once_never_raises_on_repeated_provider_failure(tmp_path, caplog):
    provider = _ScriptedProvider([RuntimeError("boom")] * 5)
    trader = PaperTrader(
        provider=provider, strategy=_ThresholdStrategy(100), backtester=_backtester(),
        log_path=tmp_path / "log.jsonl", poll_interval_seconds=3600, max_retries=5, sleep_fn=lambda s: None,
    )

    with caplog.at_level(logging.WARNING):
        record = trader.poll_once()

    assert record is None
    assert not (tmp_path / "log.jsonl").exists()
    assert trader._history is None


def test_poll_once_recovers_after_prior_tick_exhausted_retries(tmp_path):
    good_window = _make_ohlcv_at(pd.date_range("2024-01-01T00:00", periods=4, freq="h"), [90, 90, 90, 90])
    provider = _ScriptedProvider([RuntimeError("boom")] * 5 + [good_window])
    trader = PaperTrader(
        provider=provider, strategy=_ThresholdStrategy(100), backtester=_backtester(),
        log_path=tmp_path / "log.jsonl", poll_interval_seconds=3600, max_retries=5, sleep_fn=lambda s: None,
    )

    failed_tick = trader.poll_once()
    recovered_tick = trader.poll_once()

    assert failed_tick is None
    assert recovered_tick is not None


def test_run_forever_stops_after_max_iterations_without_real_sleep(tmp_path):
    ts1 = pd.date_range("2024-01-01T00:00", periods=4, freq="h")
    ts2 = pd.date_range("2024-01-01T01:00", periods=4, freq="h")
    ts3 = pd.date_range("2024-01-01T02:00", periods=4, freq="h")
    windows = [_make_ohlcv_at(ts, [90, 90, 90, 90]) for ts in (ts1, ts2, ts3)]

    provider = _ScriptedProvider(windows)
    sleep_calls = []
    trader = PaperTrader(
        provider=provider, strategy=_ThresholdStrategy(100), backtester=_backtester(),
        log_path=tmp_path / "log.jsonl", poll_interval_seconds=3600, sleep_fn=sleep_calls.append,
    )

    trader.run_forever(max_iterations=3)

    assert len(sleep_calls) == 3
    assert sleep_calls == [3600, 3600, 3600]


# --- JSONL logging --------------------------------------------------------


def test_jsonl_log_appends_one_line_per_new_bar_with_expected_fields(tmp_path):
    ts1 = pd.date_range("2024-01-01T00:00", periods=4, freq="h")
    ts2 = pd.date_range("2024-01-01T01:00", periods=4, freq="h")
    windows = [_make_ohlcv_at(ts, [90, 90, 90, 90]) for ts in (ts1, ts2)]

    provider = _ScriptedProvider(windows)
    log_path = tmp_path / "log.jsonl"
    trader = PaperTrader(
        provider=provider, strategy=_ThresholdStrategy(100), backtester=_backtester(), log_path=log_path,
        poll_interval_seconds=3600, sleep_fn=lambda s: None,
        clock_fn=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    trader.poll_once()
    trader.poll_once()

    lines = log_path.read_text().splitlines()
    assert len(lines) == 3  # 1 header + 2 decisions (poll 1: 00:00-02:00 close; poll 2: 03:00 closes)
    header = json.loads(lines[0])
    assert header["type"] == "session_start"
    assert header["symbol"] == "BTC/USDT"

    decision = json.loads(lines[1])
    assert decision["type"] == "decision"
    expected_keys = set(DecisionRecord.__dataclass_fields__) | {"type"}
    assert set(decision) == expected_keys


def test_jsonl_log_survives_across_paper_trader_instances(tmp_path):
    log_path = tmp_path / "log.jsonl"
    window1 = _make_ohlcv_at(pd.date_range("2024-01-01T00:00", periods=4, freq="h"), [90, 90, 90, 90])
    window2 = _make_ohlcv_at(pd.date_range("2024-02-01T00:00", periods=4, freq="h"), [90, 90, 90, 90])

    trader1 = PaperTrader(
        provider=_ScriptedProvider([window1]), strategy=_ThresholdStrategy(100), backtester=_backtester(),
        log_path=log_path, poll_interval_seconds=3600, sleep_fn=lambda s: None,
    )
    trader1.poll_once()
    lines_after_first = log_path.read_text().splitlines()

    trader2 = PaperTrader(
        provider=_ScriptedProvider([window2]), strategy=_ThresholdStrategy(100), backtester=_backtester(),
        log_path=log_path, poll_interval_seconds=3600, sleep_fn=lambda s: None,
    )
    trader2.poll_once()
    lines_after_second = log_path.read_text().splitlines()

    assert lines_after_second[: len(lines_after_first)] == lines_after_first
    assert len(lines_after_second) > len(lines_after_first)


def test_load_decision_log_roundtrip(tmp_path):
    log_path = tmp_path / "log.jsonl"
    append_jsonl(
        log_path,
        {
            "type": "session_start", "symbol": "BTC/USDT", "timeframe": "1h", "exchange_id": "binance",
            "strategy_name": "Threshold(100)", "strategy_registry_name": "ma_crossover",
            "strategy_params": {"fast": 20, "slow": 50}, "initial_capital": 10000.0,
            "fee": 0.001, "slippage_bps": 5.0, "started_at": "2024-01-01T00:00:00+00:00",
        },
    )
    append_jsonl(
        log_path,
        {
            "type": "decision", "polled_at": "2024-01-01T01:00:00+00:00", "bar_timestamp": "2024-01-01T00:00:00+00:00",
            "strategy_name": "Threshold(100)", "close": 100.0, "position": 0.0, "is_trade": False,
            "equity": 10000.0, "cash": 10000.0, "units": 0.0, "fee_drag": 0.0, "slippage_drag": 0.0,
        },
    )

    header, decisions = load_decision_log(log_path)

    assert header["symbol"] == "BTC/USDT"
    assert header["strategy_registry_name"] == "ma_crossover"
    assert len(decisions) == 1
    assert isinstance(decisions.index, pd.DatetimeIndex)
    assert decisions.index.name == "timestamp"
    assert decisions.iloc[0]["equity"] == 10000.0


# --- parse_timeframe_seconds -----------------------------------------------


@pytest.mark.parametrize(
    "timeframe,expected_seconds",
    [("1m", 60), ("5m", 300), ("1h", 3600), ("4h", 14400), ("1d", 86400), ("1w", 604800)],
)
def test_parse_timeframe_seconds(timeframe, expected_seconds):
    assert parse_timeframe_seconds(timeframe) == expected_seconds


def test_parse_timeframe_seconds_raises_on_invalid_unit():
    with pytest.raises(ValueError):
        parse_timeframe_seconds("1x")


def test_poll_interval_defaults_to_parsed_timeframe(tmp_path):
    provider = _ScriptedProvider([], timeframe="1h")
    trader = PaperTrader(
        provider=provider, strategy=_ThresholdStrategy(100), backtester=_backtester(), log_path=tmp_path / "log.jsonl"
    )
    assert trader.poll_interval_seconds == 3600


# --- diff_equity_paths ------------------------------------------------------


def test_diff_equity_paths_aligns_on_overlapping_timestamps_only():
    idx1 = pd.date_range("2024-01-01", periods=5, freq="h", name="timestamp")
    idx2 = pd.date_range("2024-01-01 02:00", periods=5, freq="h", name="timestamp")
    logged = pd.Series([100.0, 101, 102, 103, 104], index=idx1)
    backtested = pd.Series([100.0, 101, 102, 103, 104], index=idx2)

    diff = diff_equity_paths(logged, backtested, rebase=False)

    assert len(diff) == 3


def test_diff_equity_paths_zero_when_identical():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", name="timestamp")
    series = pd.Series([100.0, 105, 110, 108, 112], index=idx)

    diff = diff_equity_paths(series, series, rebase=False)

    assert (diff["abs_diff"] == 0).all()
    assert (diff["pct_diff"] == 0).all()


def test_diff_equity_paths_flags_known_divergence():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", name="timestamp")
    logged = pd.Series([100.0, 110.0, 90.0], index=idx)
    backtested = pd.Series([100.0, 100.0, 100.0], index=idx)

    diff = diff_equity_paths(logged, backtested, rebase=False)

    assert diff["abs_diff"].tolist() == pytest.approx([0.0, 10.0, -10.0])


def test_diff_equity_paths_rebase_collapses_constant_scale_offset():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", name="timestamp")
    backtested = pd.Series([1000.0, 1100.0, 1050.0], index=idx)
    logged = backtested * 2  # e.g. a different initial_capital, identical relative path

    rebased = diff_equity_paths(logged, backtested, rebase=True)
    raw = diff_equity_paths(logged, backtested, rebase=False)

    assert rebased["pct_diff"].abs().max() < 1e-9
    assert raw["pct_diff"].abs().max() > 0.5


def test_diff_equity_paths_raises_on_no_overlap():
    idx1 = pd.date_range("2024-01-01", periods=3, freq="h", name="timestamp")
    idx2 = pd.date_range("2025-01-01", periods=3, freq="h", name="timestamp")
    logged = pd.Series([1.0, 2.0, 3.0], index=idx1)
    backtested = pd.Series([1.0, 2.0, 3.0], index=idx2)

    with pytest.raises(ValueError):
        diff_equity_paths(logged, backtested)
