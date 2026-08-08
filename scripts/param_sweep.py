from __future__ import annotations

import pandas as pd

from _common import build_backtester, build_provider, load_config
from run_backtest import build_hard_voting_ensemble

from src.backtest.engine import Backtester
from src.evaluation.metrics import PerformanceMetrics, format_table
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.features import FeatureEngineer
from src.strategies.base import Strategy
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.registry import STRATEGY_REGISTRY


_SWEEP_FACTORIES = {**STRATEGY_REGISTRY, "hard_voting_ensemble": build_hard_voting_ensemble}

def main() -> None:
    config = load_config()
    df = FeatureEngineer().returns(DataCleaner().clean(build_provider(config).fetch()))
    metrics = PerformanceMetrics()
    sweep_cfg = config["sweep"]

    print(_ma_sweep_table(df, build_backtester(config), metrics, sweep_cfg))
    print(_min_holding_table(df, config, metrics, sweep_cfg))


def _ma_sweep_table(df: pd.DataFrame, backtester: Backtester, metrics: PerformanceMetrics, sweep_cfg: dict) -> str:
    rows = []
    for fast in sweep_cfg["fast_options"]:
        for slow in sweep_cfg["slow_options"]:
            # invalid combination
            if fast >= slow:
                continue
            row = metrics.compute(backtester.run(df, MACrossoverStrategy(fast=fast, slow=slow)))
            rows.append({**row, "fast": fast, "slow": slow})

    columns = ["fast", "slow", "total_return", "sharpe", "max_drawdown", "trade_count"]
    return format_table(columns, rows, title="\nMA crossover parameter sweep:")


def _min_holding_table(df: pd.DataFrame, config: dict, metrics: PerformanceMetrics, sweep_cfg: dict) -> str:
    rows = []
    for name in sweep_cfg["min_holding_strategies"]:
        strategy = _build_sweep_strategy(name, config)
        for min_hold in sweep_cfg["min_holding_periods"]:
            result = _backtester_with_min_hold(config, min_hold).run(df, strategy)
            row = metrics.compute(result)
            rows.append(
                {
                    "strategy": name,
                    "min_hold": min_hold,
                    "net_return": row["total_return"],
                    "sharpe": row["sharpe"],
                    "max_drawdown": row["max_drawdown"],
                    "trade_count": row["trade_count"],
                    "cost_drag": result.fee_drag.sum() + result.slippage_drag.sum(),
                }
            )

    columns = ["strategy", "min_hold", "net_return", "sharpe", "max_drawdown", "trade_count", "cost_drag"]
    return format_table(columns, rows, title="\nmin_holding_period sweep (highest-turnover strategies):")


def _build_sweep_strategy(name: str, config: dict) -> Strategy:
    params = next((entry.get("params", {}) for entry in config["strategies"] if entry["name"] == name), {})
    return _SWEEP_FACTORIES[name](**params)


def _backtester_with_min_hold(config: dict, min_holding_period: int) -> Backtester:
    return build_backtester(
        {**config, "backtest": {**config["backtest"], "min_holding_period": min_holding_period}}
    )


if __name__ == "__main__":
    main()
