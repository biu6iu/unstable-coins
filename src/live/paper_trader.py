from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from src.backtest.engine import Backtester
from src.backtest.result import BacktestResult
from src.data.base import DataProvider, validate
from src.strategies.base import Strategy

_TIMEFRAME_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_timeframe_seconds(timeframe: str) -> int:
    """Parse a ccxt-style timeframe ('1m','5m','1h','4h','1d','1w') into seconds"""
    unit = timeframe[-1]
    if unit not in _TIMEFRAME_UNIT_SECONDS:
        raise ValueError(f"Unrecognised timeframe unit in {timeframe!r}")
    return int(timeframe[:-1]) * _TIMEFRAME_UNIT_SECONDS[unit]


def fetch_with_retry(provider: DataProvider,
    max_retries: int = 5,
    base_delay_seconds: float = 2.0,
    max_delay_seconds: float = 60.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    logger: logging.Logger | None = None
) -> pd.DataFrame | None:
    """
    Call provider.fetch(), retrying with exponential backoff on failure.
    
    Returns None once retries are exhausted, so the caller can skip this tick
    """
    log = logger or logging.getLogger(__name__)
    for attempt in range(max_retries):
        try:
            return provider.fetch()
        except Exception as exc:
            log.warning("fetch failed (attempt %d/%d): %s", attempt + 1, max_retries, exc)
            if attempt < max_retries - 1:
                delay = min(base_delay_seconds * (2**attempt), max_delay_seconds)
                sleep_fn(delay)
    log.error("fetch failed after %d attempts; skipping this tick", max_retries)
    return None


@dataclass
class DecisionRecord:
    polled_at: str
    bar_timestamp: str
    strategy_name: str
    close: float
    position: float
    is_trade: bool
    equity: float
    cash: float
    units: float
    fee_drag: float
    slippage_drag: float


def append_jsonl(path: Path | str, record: dict) -> None:
    """append one JSON object as a line to `path`"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_decision_log(path: Path | str) -> tuple[dict, pd.DataFrame]:
    """read a JSONL paper-trading log back into (session_header, decisions_df)"""
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]

    headers = [r for r in records if r.get("type") == "session_start"]
    if not headers:
        raise ValueError(f"No session_start header found in {path}")

    decisions = pd.DataFrame([r for r in records if r.get("type") == "decision"])
    if not decisions.empty:
        decisions.index = pd.DatetimeIndex(pd.to_datetime(decisions["bar_timestamp"]), name="timestamp")
        decisions = decisions.sort_index()

    return headers[0], decisions


def diff_equity_paths(logged: pd.Series, backtested: pd.Series, rebase: bool = True) -> pd.DataFrame:
    """
    compare a paper-trading session's logged equity against a retrospective backtest over the same period

    rebase=True rescales both series to a common starting value so a different initial_capital between the two runs doesn't show up as a constant offset
    """
    aligned = pd.DataFrame({"logged_equity": logged, "backtested_equity": backtested}).dropna()
    if aligned.empty:
        raise ValueError("logged and backtested equity series share no overlapping timestamps")

    if rebase:
        scale = aligned["backtested_equity"].iloc[0] / aligned["logged_equity"].iloc[0]
        aligned["logged_equity"] = aligned["logged_equity"] * scale

    aligned["abs_diff"] = aligned["logged_equity"] - aligned["backtested_equity"]
    aligned["pct_diff"] = aligned["abs_diff"] / aligned["backtested_equity"]
    return aligned


class PaperTrader:

    def __init__(
        self,
        provider: DataProvider,
        strategy: Strategy,
        backtester: Backtester,
        log_path: Path | str,
        strategy_config: dict | None = None,
        poll_interval_seconds: float | None = None,
        max_retries: int = 5,
        retry_base_delay_seconds: float = 2.0,
        retry_max_delay_seconds: float = 60.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        logger: logging.Logger | None = None
    ) -> None:
        self.provider = provider
        self.strategy = strategy
        self.backtester = backtester
        self.log_path = Path(log_path)
        self.strategy_config = strategy_config
        self.poll_interval_seconds = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else parse_timeframe_seconds(self._require_timeframe(provider))
        )
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.retry_max_delay_seconds = retry_max_delay_seconds
        self.sleep_fn = sleep_fn
        self.clock_fn = clock_fn
        self.logger = logger or logging.getLogger(__name__)

        self._history: pd.DataFrame | None = None
        self._last_timestamp: pd.Timestamp | None = None
        self._session_started = False

    @staticmethod
    def _require_timeframe(provider: DataProvider) -> str:
        timeframe = getattr(provider, "timeframe", None)
        if timeframe is None:
            raise ValueError("provider has no `timeframe` attribute so pass poll_interval_seconds explicitly")
        return timeframe

    def poll_once(self) -> DecisionRecord | None:
        """fetch, detect newly closed bars, update the simulated portfolio, log, return the latest decision"""
        raw = fetch_with_retry(
            self.provider,
            max_retries=self.max_retries,
            base_delay_seconds=self.retry_base_delay_seconds,
            max_delay_seconds=self.retry_max_delay_seconds,
            sleep_fn=self.sleep_fn,
            logger=self.logger,
        )
        if raw is None:
            self.logger.warning("no data this tick (retries exhausted). Will try again next poll")
            return None

        validate(raw)
        closed = raw.iloc[:-1] 
        new_bars = self._new_bars(closed)
        if new_bars.empty:
            self.logger.info("no new closed bar since %s", self._last_timestamp)
            return None

        self._warn_if_gap(new_bars)
        history = self._merge(new_bars)

        if not self._session_started:
            self._write_session_header()
            self._session_started = True

        result = self.backtester.run(history, self.strategy)
        record = self._build_record(result)
        append_jsonl(self.log_path, {"type": "decision", **asdict(record)})
        self._print_status(record)
        return record

    def run_forever(self, max_iterations: int | None = None) -> None:
        """poll on a fixed interval matching the configured timeframe"""
        i = 0
        while max_iterations is None or i < max_iterations:
            self.poll_once()
            i += 1
            self.sleep_fn(self.poll_interval_seconds)

    def _new_bars(self, closed: pd.DataFrame) -> pd.DataFrame:
        """bars not yet seen"""
        if self._last_timestamp is None:
            return closed
        return closed[closed.index > self._last_timestamp]

    def _warn_if_gap(self, new_bars: pd.DataFrame) -> None:
        if self._last_timestamp is None:
            return
        expected_next = self._last_timestamp + pd.Timedelta(seconds=self.poll_interval_seconds)
        if new_bars.index[0] > expected_next:
            self.logger.warning(
                "possible missed bars between polls (last=%s, next closed=%s); "
                "consider a shorter poll interval or larger data.limit",
                self._last_timestamp,
                new_bars.index[0],
            )

    def _merge(self, new_bars: pd.DataFrame) -> pd.DataFrame:
        history = new_bars if self._history is None else pd.concat([self._history, new_bars])
        history = history[~history.index.duplicated(keep="last")].sort_index()
        self._history = history
        self._last_timestamp = history.index[-1]
        return history

    def _build_record(self, result: BacktestResult) -> DecisionRecord:
        position = float(result.positions.iloc[-1])
        prior_position = float(result.positions.iloc[-2]) if len(result.positions) > 1 else 0.0
        return DecisionRecord(
            polled_at=self.clock_fn().isoformat(),
            bar_timestamp=result.positions.index[-1].isoformat(),
            strategy_name=self.strategy.name,
            close=float(result.df["close"].iloc[-1]),
            position=position,
            is_trade=position != prior_position,
            equity=float(result.equity_curve.iloc[-1]),
            cash=float(result.cash.iloc[-1]),
            units=float(result.units.iloc[-1]),
            fee_drag=float(result.fee_drag.iloc[-1]),
            slippage_drag=float(result.slippage_drag.iloc[-1]),
        )

    def _write_session_header(self) -> None:
        append_jsonl(
            self.log_path,
            {
                "type": "session_start",
                "symbol": getattr(self.provider, "symbol", None),
                "timeframe": getattr(self.provider, "timeframe", None),
                "exchange_id": getattr(self.provider, "exchange_id", None),
                "strategy_name": self.strategy.name,
                "strategy_registry_name": (self.strategy_config or {}).get("name"),
                "strategy_params": (self.strategy_config or {}).get("params", {}),
                "initial_capital": self.backtester.initial_capital,
                "fee": self.backtester.fee,
                "slippage_bps": self.backtester.slippage_bps,
                "started_at": self.clock_fn().isoformat(),
            },
        )

    def _print_status(self, record: DecisionRecord) -> None:
        trade_flag = " (TRADE)" if record.is_trade else ""
        print(
            f"[{record.bar_timestamp}] {record.strategy_name}: "
            f"position={record.position:.2f} close={record.close:.2f} "
            f"equity={record.equity:,.2f}{trade_flag}"
        )
