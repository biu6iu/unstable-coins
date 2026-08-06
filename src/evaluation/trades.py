"""
Derives discrete round-trip trades from a BacktestResult's position
series, and summary P/L statistics from those trades. The engine only
tracks an aggregate trade_count; this module is where individual
entries/exits are reconstructed for a per-trade log.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.result import BacktestResult


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    holding_bars: int
    pnl_dollars: float
    pnl_pct: float
    fees_paid: float
    slippage_paid: float


def extract_trades(result: BacktestResult) -> list[Trade]:
    """
    A trade is one contiguous span of bars where `positions != 0`
    (entry to exit). Boundaries follow the engine's one-bar execution
    delay: position[t] earns bar t's return, having been decided by
    signal[t-1], so a trade entered at bar `entry_idx` is priced off
    close[entry_idx - 1] (the reference the engine's own return math
    uses), and pnl_dollars spans equity_curve[entry_idx-1] through
    equity_curve[exit_idx] so it picks up BOTH the entry fee (charged
    at entry_idx) and the exit fee (charged at exit_idx, the first
    flat bar after the run).

    A position still open at the end of the data (no matching exit
    boundary) is deliberately excluded - this is a log of CLOSED
    trades only.
    """
    positions = result.positions.to_numpy()
    in_market = (positions != 0).astype(int)
    transitions = np.diff(in_market, prepend=0)
    entry_indices = np.flatnonzero(transitions == 1)
    exit_indices = np.flatnonzero(transitions == -1)

    n_closed = min(len(entry_indices), len(exit_indices))
    trades = []
    for entry_idx, exit_idx in zip(entry_indices[:n_closed], exit_indices[:n_closed]):
        trades.append(_build_trade(result, int(entry_idx), int(exit_idx)))
    return trades


def _build_trade(result: BacktestResult, entry_idx: int, exit_idx: int) -> Trade:
    close = result.df["close"]
    equity = result.equity_curve

    entry_equity = equity.iloc[entry_idx - 1]
    pnl_dollars = float(equity.iloc[exit_idx] - entry_equity)

    return Trade(
        entry_time=result.df.index[entry_idx],
        exit_time=result.df.index[exit_idx - 1],
        entry_price=float(close.iloc[entry_idx - 1]),
        exit_price=float(close.iloc[exit_idx - 1]),
        holding_bars=exit_idx - entry_idx,
        pnl_dollars=pnl_dollars,
        pnl_pct=pnl_dollars / float(entry_equity),
        fees_paid=float(result.fee_drag.iloc[entry_idx : exit_idx + 1].sum()),
        slippage_paid=float(result.slippage_drag.iloc[entry_idx : exit_idx + 1].sum()),
    )


def trade_summary(trades: list[Trade]) -> dict:
    """Win rate, average win/loss, profit factor, and total P/L across a list of trades"""
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": math.nan,
            "avg_win": math.nan,
            "avg_loss": math.nan,
            "profit_factor": math.nan,
            "total_pnl_dollars": 0.0,
            "total_fees_paid": 0.0,
            "total_slippage_paid": 0.0,
        }

    wins = [t for t in trades if t.pnl_dollars > 0]
    losses = [t for t in trades if t.pnl_dollars <= 0]
    gross_profit = sum(t.pnl_dollars for t in wins)
    gross_loss = -sum(t.pnl_dollars for t in losses)  # positive magnitude

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = math.inf if gross_profit > 0 else math.nan

    return {
        "num_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_win": gross_profit / len(wins) if wins else math.nan,
        "avg_loss": -gross_loss / len(losses) if losses else math.nan,
        "profit_factor": profit_factor,
        "total_pnl_dollars": sum(t.pnl_dollars for t in trades),
        "total_fees_paid": sum(t.fees_paid for t in trades),
        "total_slippage_paid": sum(t.slippage_paid for t in trades),
    }
