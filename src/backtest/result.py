from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    """
    cash and units are a reconciliation view back-solved from equity_curve
    (see src.backtest.engine.ledger_fields), not an independent
    forward-simulated ledger; equity_curve itself is still built from
    strategy_returns = gross_returns - fee_drag - slippage_drag.
    """

    df: pd.DataFrame
    positions: pd.Series
    strategy_returns: pd.Series
    equity_curve: pd.Series
    trade_count: int
    strategy_name: str
    gross_returns: pd.Series
    fee_drag: pd.Series
    slippage_drag: pd.Series
    cash: pd.Series
    units: pd.Series