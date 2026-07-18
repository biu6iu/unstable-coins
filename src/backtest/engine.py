from __future__ import annotations
import pandas as pd

from src.backtest.result import BacktestResult
from src.strategies.base import Strategy


class Backtester:

    def __init__(self, fee: float = 0.001, initial_capital: float = 10000.0) -> None:
        self.fee = fee
        self.initial_capital = initial_capital

    def run(self, df: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        signals = strategy.generate_signals(df)

        # prevent lookahead bias
        position = signals["signal"].shift(1).fillna(0)
        asset_returns = signals["close"].pct_change().fillna(0)

        # identify all position changes and count trades to prevent whipsawing
        position_change = position.diff().fillna(0).abs()
        trade_count = int((position_change > 0).sum())

        # calculate returns
        strategy_returns = position * asset_returns - position_change * self.fee
        equity_curve = self.initial_capital * (1 + strategy_returns).cumprod()

        return BacktestResult(
            df=df,
            positions=position,
            strategy_returns=strategy_returns,
            equity_curve=equity_curve,
            trade_count=trade_count,
            strategy_name=strategy.name,
        )
