from __future__ import annotations
import numpy as np
import pandas as pd

from src.backtest.result import BacktestResult
from src.strategies.base import Strategy


class Backtester:

    def __init__(
        self,
        fee: float = 0.001,
        initial_capital: float = 10000.0,
        slippage_bps: float = 5.0,
        min_holding_period: int = 0,
    ) -> None:
        self.fee = fee
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps
        self.min_holding_period = min_holding_period

    def run(self, df: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        signals = strategy.generate_signals(df)

        # prevent lookahead bias
        raw_position = signals["signal"].shift(1).fillna(0)
        position = _apply_min_holding_period(raw_position, self.min_holding_period)
        asset_returns = signals["close"].pct_change().fillna(0)

        # identify all position changes and count trades to prevent whipsawing
        position_change = position.diff().fillna(0).abs()
        trade_count = int((position_change > 0).sum())

        # gross_returns is what the position would have earned with zero costs
        gross_returns = position * asset_returns

        # fee_drag is the cost of rebalancing to the new position, charged on the notional traded
        fee_drag = position_change * self.fee

        # slippage_drag is the adverse price impact of that same rebalancing 
        slippage_drag = position_change * (self.slippage_bps / 10_000)

        strategy_returns = gross_returns - fee_drag - slippage_drag
        equity_curve = self.initial_capital * (1 + strategy_returns).cumprod()

        equity_prior = equity_curve.shift(1).fillna(self.initial_capital)
        prior_close = signals["close"].shift(1)
        units = (position * equity_prior / prior_close).fillna(0)

        cash = equity_prior * (1 - position - fee_drag - slippage_drag)

        return BacktestResult(
            df=df,
            positions=position,
            strategy_returns=strategy_returns,
            equity_curve=equity_curve,
            trade_count=trade_count,
            strategy_name=strategy.name,
            gross_returns=gross_returns,
            fee_drag=fee_drag,
            slippage_drag=slippage_drag,
            cash=cash,
            units=units,
        )


def _apply_min_holding_period(position: pd.Series, min_holding_period: int) -> pd.Series:
    """Suppress position flips until `min_holding_period` bars have elapsed since the last accepted flip """
    if min_holding_period <= 1:
        return position

    values = position.to_numpy()
    held = np.empty_like(values)
    held[0] = values[0]
    last_change_bar = 0
    for i in range(1, len(values)):
        if values[i] != held[i - 1] and (i - last_change_bar) >= min_holding_period:
            held[i] = values[i]
            last_change_bar = i
        else:
            held[i] = held[i - 1]
    return pd.Series(held, index=position.index)
