from __future__ import annotations
import numpy as np

from src.backtest.result import BacktestResult

ANNUALISATION_FACTOR = 365

_COLUMNS = [
    "strategy",
    "total_return",
    "cagr",
    "annualised_volatility",
    "sharpe",
    "max_drawdown",
    "trade_count",
]

class PerformanceMetrics:

    def compute(self, result: BacktestResult) -> dict:
        """Manually compute total/annualised return, volatility, Sharpe, max drawdown, and trade count for one backtest run"""
        equity = result.equity_curve
        returns = result.strategy_returns

        total_return = equity.iloc[-1] / equity.iloc[0] - 1

        years = len(equity) / ANNUALISATION_FACTOR
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan

        annualised_volatility = returns.std() * np.sqrt(ANNUALISATION_FACTOR)

        sharpe = (
            (returns.mean() / returns.std()) * np.sqrt(ANNUALISATION_FACTOR)
            if returns.std() > 0
            else np.nan
        )

        running_max = equity.cummax()
        max_drawdown = (equity / running_max - 1).min()

        return {
            "strategy": result.strategy_name,
            "total_return": total_return,
            "cagr": cagr,
            "annualised_volatility": annualised_volatility,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "trade_count": result.trade_count,
        }

    def compare(self, results: list[BacktestResult]) -> str:
        """Print and return a formatted side-by-side table for multiple
        strategies, each computed with `compute`."""
        rows = [self.compute(result) for result in results]

        widths = {
            column: max(len(column), *(len(_format(row[column])) for row in rows))
            for column in _COLUMNS
        }
        header = " | ".join(column.ljust(widths[column]) for column in _COLUMNS)
        separator = "-+-".join("-" * widths[column] for column in _COLUMNS)
        body = [
            " | ".join(_format(row[column]).ljust(widths[column]) for column in _COLUMNS)
            for row in rows
        ]

        table = "\n".join([header, separator, *body])
        print(table)
        return table


def _format(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
