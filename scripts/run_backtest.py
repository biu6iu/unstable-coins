from __future__ import annotations
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.analysis.monte_carlo import MonteCarloAnalyzer
from src.backtest.engine import Backtester
from src.backtest.result import BacktestResult
from src.data.ccxt_provider import CCXTDataProvider
from src.data.synthetic import SyntheticDataProvider
from src.evaluation.metrics import PerformanceMetrics
from src.evaluation.plots import ReportPlotter
from src.pipeline import Pipeline
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.features import FeatureEngineer
from src.strategies.buy_and_hold import BuyAndHoldStrategy
from src.strategies.ma_crossover import MACrossoverStrategy

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/config.yaml")

def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _build_provider(config: dict):
    data_cfg = config["data"]
    ccxt_provider = CCXTDataProvider(
        symbol=data_cfg["symbol"],
        timeframe=data_cfg["timeframe"],
        limit=data_cfg["limit"],
    )
    try:
        ccxt_provider.fetch() 
        return ccxt_provider
    except Exception as exc:
        logger.warning("CCXT fetch failed (%s); falling back to synthetic data", exc)
        return SyntheticDataProvider(n_periods=data_cfg["limit"])


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = _load_config()

    provider = _build_provider(config)

    backtest_cfg = config["backtest"]
    backtester = Backtester(
        fee=backtest_cfg["fee"],
        initial_capital=backtest_cfg["initial_capital"],
        slippage_bps=backtest_cfg.get("slippage_bps", 5.0),
        min_holding_period=backtest_cfg.get("min_holding_period", 0),
    )

    strategy_cfg = config["strategy"]
    strategies = [
        MACrossoverStrategy(fast=strategy_cfg["fast_ma"], slow=strategy_cfg["slow_ma"]),
        BuyAndHoldStrategy(),
    ]

    mc_cfg = config.get("monte_carlo", {})
    monte_carlo = (
        MonteCarloAnalyzer(
            n_trials=mc_cfg.get("n_trials", 5000),
            block_length=mc_cfg.get("block_length", 20),
            noise_std=mc_cfg.get("noise_std", 0.001),
        )
        if mc_cfg.get("enabled", False)
        else None
    )

    pipeline = Pipeline(
        provider=provider,
        cleaner=DataCleaner(),
        engineer=FeatureEngineer(),
        strategies=strategies,
        backtester=backtester,
        metrics=PerformanceMetrics(),
        plotter=ReportPlotter(),
        monte_carlo=monte_carlo,
    )

    results = pipeline.run(plot_filename="ma_crossover_vs_buy_and_hold.png")
    _print_cost_impact_table(results)


def _print_cost_impact_table(results: list[BacktestResult]) -> None:
    columns = ["strategy", "gross_return", "fee_drag", "slippage_drag", "net_return"]
    rows = []
    for result in results:
        gross_return = (1 + result.gross_returns).prod() - 1
        net_return = result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1
        rows.append(
            {
                "strategy": result.strategy_name,
                "gross_return": gross_return,
                "fee_drag": result.fee_drag.sum(),
                "slippage_drag": result.slippage_drag.sum(),
                "net_return": net_return,
            }
        )

    widths = {c: max(len(c), *(len(f"{row[c]:.4f}" if c != "strategy" else row[c]) for row in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    print("\nCost impact (before/after fees + slippage):")
    print(header)
    print("-+-".join("-" * widths[c] for c in columns))
    for row in rows:
        cells = [
            row["strategy"].ljust(widths["strategy"]),
            *(f"{row[c]:.4f}".ljust(widths[c]) for c in columns[1:]),
        ]
        print(" | ".join(cells))


if __name__ == "__main__":
    main()
