import math

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Backtester
from src.evaluation.trades import Trade, extract_trades, trade_summary
from src.strategies.base import Strategy


def _make_ohlcv(close):
    close = np.asarray(close, dtype=float)
    index = pd.date_range("2022-01-01", periods=len(close), freq="D", name="timestamp")
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(len(close), 100.0),
        },
        index=index,
    )


class _FixedSignalStrategy(Strategy):
    def __init__(self, signal):
        self._signal = signal

    @property
    def name(self) -> str:
        return "Fixed"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = self._signal
        return out


def _run(close, signal, fee=0.001, initial_capital=1000.0, slippage_bps=5.0):
    df = _make_ohlcv(close)
    strategy = _FixedSignalStrategy(signal)
    backtester = Backtester(fee=fee, initial_capital=initial_capital, slippage_bps=slippage_bps)
    return backtester.run(df, strategy)


# --- extract_trades ---------------------------------------------------


def test_extract_trades_single_closed_trade_matches_hand_computed_values():
    close = [100, 110, 90, 130, 120, 140]
    result = _run(close, signal=[1, 1, 0, 0, 0, 0])

    trades = extract_trades(result)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.entry_time == result.df.index[1]
    assert trade.exit_time == result.df.index[2]
    assert trade.entry_price == pytest.approx(100.0)
    assert trade.exit_price == pytest.approx(90.0)
    assert trade.holding_bars == 2
    assert trade.pnl_dollars == pytest.approx(
        result.equity_curve.iloc[3] - result.equity_curve.iloc[0]
    )
    assert trade.pnl_pct == pytest.approx(trade.pnl_dollars / result.equity_curve.iloc[0])
    assert trade.fees_paid == pytest.approx(result.fee_drag.iloc[1] + result.fee_drag.iloc[3])
    assert trade.slippage_paid == pytest.approx(
        result.slippage_drag.iloc[1] + result.slippage_drag.iloc[3]
    )


def test_extract_trades_excludes_still_open_trailing_position():
    close = [100, 110, 90, 130, 120, 140]
    result = _run(close, signal=[1, 1, 1, 1, 1, 1])  # never exits

    assert extract_trades(result) == []


def test_extract_trades_returns_empty_list_when_never_in_market():
    close = [100, 110, 90, 130, 120, 140]
    result = _run(close, signal=[0, 0, 0, 0, 0, 0])

    assert extract_trades(result) == []


def test_extract_trades_finds_multiple_separate_round_trips():
    close = [100, 110, 90, 130, 120, 140, 150, 160, 170, 180]
    signal = [1, 1, 0, 0, 1, 1, 0, 0, 0, 0]
    result = _run(close, signal=signal)

    trades = extract_trades(result)

    assert len(trades) == 2
    assert trades[0].entry_time == result.df.index[1]
    assert trades[0].exit_time == result.df.index[2]
    assert trades[1].entry_time == result.df.index[5]
    assert trades[1].exit_time == result.df.index[6]


# --- trade_summary ------------------------------------------------------


def _trade(pnl_dollars: float, fees: float = 0.0, slippage: float = 0.0) -> Trade:
    return Trade(
        entry_time=pd.Timestamp("2022-01-01"),
        exit_time=pd.Timestamp("2022-01-02"),
        entry_price=100.0,
        exit_price=100.0 + pnl_dollars,
        holding_bars=1,
        pnl_dollars=pnl_dollars,
        pnl_pct=pnl_dollars / 100.0,
        fees_paid=fees,
        slippage_paid=slippage,
    )


def test_trade_summary_empty_list():
    summary = trade_summary([])

    assert summary["num_trades"] == 0
    assert math.isnan(summary["win_rate"])
    assert math.isnan(summary["avg_win"])
    assert math.isnan(summary["avg_loss"])
    assert math.isnan(summary["profit_factor"])
    assert summary["total_pnl_dollars"] == 0.0


def test_trade_summary_all_wins_has_infinite_profit_factor():
    summary = trade_summary([_trade(10.0), _trade(20.0), _trade(30.0)])

    assert summary["win_rate"] == 1.0
    assert summary["avg_win"] == pytest.approx(20.0)
    assert math.isnan(summary["avg_loss"])
    assert summary["profit_factor"] == math.inf
    assert summary["total_pnl_dollars"] == pytest.approx(60.0)


def test_trade_summary_all_losses_has_zero_profit_factor():
    summary = trade_summary([_trade(-10.0), _trade(-20.0)])

    assert summary["win_rate"] == 0.0
    assert math.isnan(summary["avg_win"])
    assert summary["avg_loss"] == pytest.approx(-15.0)
    assert summary["profit_factor"] == pytest.approx(0.0)
    assert summary["total_pnl_dollars"] == pytest.approx(-30.0)


def test_trade_summary_mixed_wins_and_losses():
    summary = trade_summary([_trade(10.0), _trade(-5.0), _trade(20.0), _trade(-15.0)])

    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["avg_win"] == pytest.approx(15.0)
    assert summary["avg_loss"] == pytest.approx(-10.0)
    assert summary["profit_factor"] == pytest.approx(1.5)
    assert summary["total_pnl_dollars"] == pytest.approx(10.0)


def test_trade_summary_zero_pnl_trade_counts_as_a_loss_not_a_win():
    summary = trade_summary([_trade(0.0)])

    assert summary["win_rate"] == 0.0
    assert math.isnan(summary["avg_win"])
    assert summary["avg_loss"] == pytest.approx(0.0)
    assert math.isnan(summary["profit_factor"])


def test_trade_summary_totals_fees_and_slippage():
    summary = trade_summary([_trade(10.0, fees=0.5, slippage=0.25), _trade(-5.0, fees=0.5, slippage=0.25)])

    assert summary["total_fees_paid"] == pytest.approx(1.0)
    assert summary["total_slippage_paid"] == pytest.approx(0.5)
