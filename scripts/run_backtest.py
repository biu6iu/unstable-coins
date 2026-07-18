from __future__ import annotations

import logging
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.engine import Backtester
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
        fee=backtest_cfg["fee"], initial_capital=backtest_cfg["initial_capital"]
    )

    strategy_cfg = config["strategy"]
    strategies = [
        MACrossoverStrategy(fast=strategy_cfg["fast_ma"], slow=strategy_cfg["slow_ma"]),
        BuyAndHoldStrategy(),
    ]

    pipeline = Pipeline(
        provider=provider,
        cleaner=DataCleaner(),
        engineer=FeatureEngineer(),
        strategies=strategies,
        backtester=backtester,
        metrics=PerformanceMetrics(),
        plotter=ReportPlotter(),
    )
    
    pipeline.run(plot_filename="ma_crossover_vs_buy_and_hold.png")


if __name__ == "__main__":
    main()
