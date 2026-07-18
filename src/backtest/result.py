from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    df: pd.DataFrame
    positions: pd.Series
    strategy_returns: pd.Series
    equity_curve: pd.Series
    trade_count: int
    strategy_name: str