from __future__ import annotations
import numpy as np
import pandas as pd

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

def max_drawdown(equity: pd.Series) -> float:
    """Peak-to-trough decline of an equity curve, as a fraction (e.g. -0.25 = 25% drawdown)"""
    running_max = equity.cummax()
    return (equity / running_max - 1).min()


def sharpe_ratio(returns: pd.Series) -> float:
    """
    Annualised Sharpe at a zero risk-free rate
    """
    std = returns.std()
    if pd.isna(std) or std == 0:
        return np.nan
    return (returns.mean() / std) * np.sqrt(ANNUALISATION_FACTOR)


class PerformanceMetrics:
    def compute(self, result: BacktestResult) -> dict:
        """Manually compute total/annualised return, volatility, Sharpe, max drawdown, and trade count for one backtest run"""
        equity = result.equity_curve
        returns = result.strategy_returns

        total_return = equity.iloc[-1] / equity.iloc[0] - 1

        years = len(equity) / ANNUALISATION_FACTOR
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan

        annualised_volatility = returns.std() * np.sqrt(ANNUALISATION_FACTOR)
        sharpe = sharpe_ratio(returns)

        return {
            "strategy": result.strategy_name,
            "total_return": total_return,
            "cagr": cagr,
            "annualised_volatility": annualised_volatility,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown(equity),
            "trade_count": result.trade_count,
        }

    def compare(self, results: list[BacktestResult]) -> str:
        """Print and return a formatted side-by-side table for multiple
        strategies, each computed with `compute`."""
        rows = [self.compute(result) for result in results]
        table = format_table(_COLUMNS, rows)
        print(table)
        return table


def _format(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_table(columns: list[str], rows: list[dict], title: str | None = None) -> str:
    """Render `rows` (dicts keyed by `columns`) as an aligned text table; floats shown to 4dp"""
    widths = {
        column: max(len(column), *(len(_format(row[column])) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(_format(row[column]).ljust(widths[column]) for column in columns)
        for row in rows
    ]
    lines = ([title] if title else []) + [header, separator, *body]
    return "\n".join(lines)
