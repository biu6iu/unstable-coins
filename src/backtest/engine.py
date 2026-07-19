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

        # gross_returns is what the position would have earned with zero costs
        # fee_drag is the cost of rebalancing to the new position, charged on the notional traded
        gross_returns = position * asset_returns
        fee_drag = position_change * self.fee
        strategy_returns = gross_returns - fee_drag
        equity_curve = self.initial_capital * (1 + strategy_returns).cumprod()

        equity_prior = equity_curve.shift(1).fillna(self.initial_capital)
        prior_close = signals["close"].shift(1)
        units = (position * equity_prior / prior_close).fillna(0)
    
        cash = equity_prior * (1 - position - fee_drag)

        return BacktestResult(
            df=df,
            positions=position,
            strategy_returns=strategy_returns,
            equity_curve=equity_curve,
            trade_count=trade_count,
            strategy_name=strategy.name,
            gross_returns=gross_returns,
            fee_drag=fee_drag,
            cash=cash,
            units=units,
        )
